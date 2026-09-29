"""Four independent GPU workers; aggregate STFlow records and FVD sufficient statistics."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
from unittest.mock import patch

ROOT = Path('/inspire/qb-ilm/project/quantum-artificial-intelligence/yanjunchi-24040/songbur/camsim')
EVAL = ROOT / 'lyh_output/eval/nuplanhard1000'
WEIGHTS = ROOT.parent / 'pretrain/ckpt'


def write_stable(path, text):
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f'Existing shard/config differs: {path}; use a new output directory')
        return
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(text)
    temporary.replace(path)


def prepare(args):
    # Stream manifests; do not retain all box annotations in memory.
    manifests = []
    fingerprints = []
    ids = []
    poses = set()
    missing = 0
    for rank in range(4):
        manifests.append([])
    with args.manifest.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            index = len(ids)
            if index >= args.max_videos:
                break
            item = json.loads(line)
            expected_id = f'nuplan_video_{index:06d}'
            if item['video_id'] != expected_id:
                raise RuntimeError(f'Unexpected merged order at {index}')
            ids.append(expected_id)
            if [f['frame_index'] for f in item['frames']] != list(range(19)):
                raise RuntimeError(f'Invalid time order: {expected_id}')
            pose = json.dumps([f['T_ego_to_world'] for f in item['frames']])
            if pose in poses:
                raise RuntimeError(f'Duplicate trajectory: {expected_id}')
            poses.add(pose)
            for frame in item['frames']:
                if [v['camera'] for v in frame['views']] != [f'CAM_{i:02d}' for i in range(8)]:
                    raise RuntimeError(f'Invalid camera order: {expected_id}')
                for view in frame['views']:
                    for key in ('image_path', 'real_image_path', 'valid_mask_path'):
                        if view.get(key) is None and key == 'valid_mask_path':
                            continue
                        path = (args.manifest.parent / view[key]).resolve()
                        missing += not path.is_file()
                        view[key] = str(path)
            text = json.dumps(item, ensure_ascii=False) + '\n'
            fingerprints.append(hashlib.sha256(text.encode()).hexdigest())
            manifests[index % 4].append(text)
    if len(ids) != args.max_videos or missing:
        raise RuntimeError(f'Expected {args.max_videos} videos; found {len(ids)}; missing files={missing}')
    config = {'manifest': str(args.manifest.resolve()), 'ids': ids, 'fingerprints': fingerprints,
              'fvd_batch_size': args.batch_size, 'weights': str(WEIGHTS)}
    write_stable(args.output_dir / 'inputs.json', json.dumps(config, indent=2))
    for rank, lines in enumerate(manifests):
        shard = args.output_dir / f'rank_{rank:02d}'
        shard.mkdir(exist_ok=True)
        write_stable(shard / 'stflow_manifest.jsonl', ''.join(lines))
    print(f'[CHECK] {len(ids)} unique videos; shards={[len(x) for x in manifests]}; missing files=0', flush=True)
    return ids


def worker(args):
    shard = args.output_dir / f'rank_{args.worker:02d}'
    manifest = str(shard / 'stflow_manifest.jsonl')
    # Block downloads in both evaluators; only prepared local weights are allowed.
    with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('Network download disabled; check local weights')):
        sys.argv = ['evaluate_stflow', '--manifest', manifest, '--output', str(shard / 'stflow.json'),
                    '--device', 'cuda', '--frame-stride', '2', '--min-matches', '16',
                    '--max-matches', '256', '--loftr-confidence', '0.1', '--pair-policy', 'dataset', '--cross-gate-px', '16']
        runpy.run_module('dwm.tools.evaluate_stflow', run_name='__main__')
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
        sys.argv = ['evaluate_fvd', '--manifest', manifest, '--output', str(shard / 'fvd.json'),
                    '--i3d-checkpoint', str(WEIGHTS / 'i3d_pretrained_400.pt'), '--device', 'cuda',
                    '--sequence-count', '19', '--batch-size', str(args.batch_size)]
        runpy.run_module('dwm.tools.evaluate_fvd_from_paired_manifest', run_name='__main__')


def combine(args, ids):
    import torch
    from dwm.tools.evaluate_stflow import aggregate_video_results
    from dwm.tools.evaluate_fvd_from_paired_manifest import FVD_STATE_NAMES, write_json
    from dwm.metrics.fvd import _compute_fid
    results = {}
    total = {}
    eval_config = None
    for rank in range(4):
        shard = args.output_dir / f'rank_{rank:02d}'
        stflow = json.loads((shard / 'stflow.json').read_text())
        expected = ids[rank::4]
        if [r['video_id'] for r in stflow['videos']] != expected:
            raise RuntimeError(f'STFlow shard {rank} incomplete or misordered')
        if eval_config is not None and stflow['eval_config'] != eval_config:
            raise RuntimeError('STFlow configuration mismatch')
        eval_config = stflow['eval_config']
        results.update({r['video_id']: r for r in stflow['videos']})
        checkpoint = torch.load(shard / 'fvd.json.progress.pt', map_location='cpu', weights_only=True)
        signature = checkpoint['signature']
        expected_signature = {'manifest': str((shard / 'stflow_manifest.jsonl').resolve()),
                              'i3d_checkpoint': str((WEIGHTS / 'i3d_pretrained_400.pt').resolve()),
                              'camera_names': [f'CAM_{i:02d}' for i in range(8)], 'sequence_count': 19,
                              'batch_size': args.batch_size, 'num_samples': len(expected) * 8}
        if signature != expected_signature or checkpoint['next_start'] != len(expected) * 8:
            raise RuntimeError(f'FVD shard {rank} incomplete or configuration mismatch')
        state = checkpoint['metric_state']
        for prefix in ('real', 'fake'):
            if int(state[prefix + '_features_num_samples']) != len(expected) * 8:
                raise RuntimeError('FVD sample count mismatch')
        for key in FVD_STATE_NAMES:
            total[key] = total.get(key, 0) + state[key]
    means, covariances = [], []
    for prefix in ('real', 'fake'):
        n = total[prefix + '_features_num_samples']
        mean = total[prefix + '_features_sum'] / n
        means.append(mean)
        covariances.append((total[prefix + '_features_cov_sum'] - n * torch.outer(mean, mean)) / (n - 1))
    value = float(_compute_fid(means[0], covariances[0], means[1], covariances[1]))
    result = aggregate_video_results([results[key] for key in ids])
    result['eval_config'] = eval_config
    write_json(args.output_dir / 'stflow_traj_result_gate16.json', result)
    write_json(args.output_dir / 'paired_fvd_result_all19.json', {
        'fvd': value, 'num_videos': len(ids), 'num_samples': len(ids) * 8, 'sequence_count': 19,
        'camera_names': [f'CAM_{i:02d}' for i in range(8)], 'batch_size': args.batch_size,
        'manifest': str(args.manifest), 'aggregation': 'sum feature statistics across 4 shards'})
    print(f'[DONE] FVD={value}; results={args.output_dir}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=EVAL / 'bev_pv_epipolar32000_merged1000/stflow_manifest.jsonl')
    parser.add_argument('--output-dir', type=Path, default=EVAL / 'bev_pv_epipolar32000_merged1000/eval_4gpu')
    parser.add_argument('--gpus', default=os.environ.get('CUDA_VISIBLE_DEVICES', '0,1,2,3'))
    parser.add_argument('--max-videos', type=int, default=1000)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--worker', type=int, choices=range(4), help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker is not None:
        worker(args)
        return
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != 4 or not all(gpus) or not 4 <= args.max_videos <= 1000 or args.batch_size < 1:
        parser.error('Select 4 distinct GPUs, 4..1000 videos, and positive batch size')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        lock = stack.enter_context((args.output_dir / 'evaluation.lock').open('a'))
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cache = args.output_dir / 'torch_cache/hub/checkpoints'
        cache.mkdir(parents=True, exist_ok=True)
        for name in ('loftr_outdoor.ckpt', 'raft_large_C_T_SKHT_V2-ff5fadd5.pth', 'i3d_pretrained_400.pt'):
            if not (WEIGHTS / name).is_file():
                raise FileNotFoundError(WEIGHTS / name)
        shutil.copyfile(WEIGHTS / 'loftr_outdoor.ckpt', cache / 'loftr_outdoor.ckpt')
        os.environ['TORCH_HOME'] = str(args.output_dir / 'torch_cache')
        os.environ['DWM_RAFT_WEIGHTS'] = str(WEIGHTS / 'raft_large_C_T_SKHT_V2-ff5fadd5.pth')
        import torch
        from dwm.metrics.stflow import STFlowEvaluator
        from dwm.metrics.fvd import FrechetVideoDistance
        with patch('torch.hub.download_url_to_file', side_effect=RuntimeError('Offline weights missing')):
            evaluator = STFlowEvaluator(device='cpu')
            fvd = FrechetVideoDistance(str(WEIGHTS / 'i3d_pretrained_400.pt'), sequence_count=19)
        del evaluator, fvd
        print('[CHECK] RAFT, LoFTR, I3D loaded locally with downloads disabled', flush=True)
        ids = prepare(args)
        if args.check_only:
            print('[CHECK] Passed; no GPU evaluation launched', flush=True)
            return
        processes = []
        try:
            for rank, gpu in enumerate(gpus):
                log = stack.enter_context((args.output_dir / f'rank_{rank:02d}/worker.log').open('a'))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
                command = [sys.executable, str(Path(__file__).resolve()), '--worker', str(rank),
                           '--output-dir', str(args.output_dir), '--batch-size', str(args.batch_size)]
                processes.append(subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT))
            codes = [process.wait() for process in processes]
            if any(codes):
                raise RuntimeError(f'Workers failed: {codes}; inspect rank_*/worker.log; rerun same command to resume')
        finally:
            for process in processes:
                if process.poll() is None:
                    process.terminate()
            for process in processes:
                process.wait()
        combine(args, ids)


if __name__ == '__main__':
    main()
