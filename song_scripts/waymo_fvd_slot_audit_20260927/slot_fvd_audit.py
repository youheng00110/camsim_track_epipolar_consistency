"""Standalone Waymo camera-slot sensitivity experiment; original metric is imported unchanged."""
import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import numpy as np

METHODS = {
    'plucker': 'plucker_6hz_merged1000',
    'implicit': 'implicit_6hz_18000_1000_merged1000',
    'petr_correct': 'petr_6hz_18000_1000_original_0e57bb3b_merged1000',
    'box': 'box_6hz_18000_1000_merged1000',
    'pvonly': 'pvonly_6hz_merged1000',
}
PAIRS = [(1,6),(2,5),(3,7)]


def moments(features, camera_weights):
    # Frequency-mass convention: each physical camera contributes one unit per video.
    # Thus exact duplicate collapse equals the five-camera unbiased sample covariance.
    x = np.asarray(features, dtype=np.float64).reshape(-1, features.shape[-1])
    w = np.tile(np.asarray(camera_weights, dtype=np.float64), features.shape[0])
    total = w.sum()
    if total <= 1:
        raise ValueError('At least two samples required')
    mean = (x * w[:, None]).sum(axis=0) / total
    centered = x - mean
    covariance = (centered * w[:, None]).T @ centered / (total - 1)
    return mean, covariance


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    os.replace(tmp, path)


def sha_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


class VideoPairs:
    def __init__(self, manifest, rows):
        self.root = str(manifest.parent)
        self.rows = rows

    def __len__(self):
        return len(self.rows) * 8

    def __getitem__(self, index):
        from dwm.tools.evaluate_fvd_from_paired_manifest import load_video_pair
        fake, real = load_video_pair(self.root, self.rows[index//8], f'CAM_{index%8:02d}', 19)
        # Hash decoded generated-only pixels, not compressed JPG bytes or reference frames.
        fake_hash = hashlib.sha256((fake[3:] * 255).round().byte().numpy().tobytes()).hexdigest()
        real_hash = hashlib.sha256((real[3:] * 255).round().byte().numpy().tobytes()).hexdigest()
        return fake, real, fake_hash, real_hash


class FeatureTap:
    def __init__(self):
        self.value = None

    def __call__(self, module, inputs, output):
        self.value = output.detach().cpu().numpy().copy()


def preflight(root, output, limit):
    baseline = None
    for method, folder in METHODS.items():
        manifest = root / folder / 'stflow_manifest.jsonl'
        fingerprints = []
        with manifest.open() as stream:
            lines = itertools.islice((line for line in stream if line.strip()), limit)
            for line in lines:
                fingerprints.append(audit_row(json.loads(line), method))
        if len(fingerprints) != limit:
            raise ValueError(f'{method}: expected {limit} videos')
        if baseline is None:
            baseline = fingerprints
        if fingerprints != baseline or len(set(fingerprints)) != limit:
            raise ValueError(f'{method}: sample alignment or duplication error')
        print(f'[PREFLIGHT] {method} {limit} unique and aligned videos', flush=True)
    atomic_json(output/'preflight.json', {'videos':limit,'methods':METHODS,'geometry_sha256':baseline,'pairs':PAIRS})


def audit_row(row, method):
    if [f['frame_index'] for f in row['frames']] != list(range(19)):
        raise ValueError(f'{method}: frame order mismatch')
    geometry = []
    for frame in row['frames']:
        vm = {v['camera']: v for v in frame['views']}
        if sorted(vm) != [f'CAM_{c:02d}' for c in range(8)]:
            raise ValueError('camera set mismatch')
        for a,b in PAIRS:
            va,vb = vm[f'CAM_{a:02d}'],vm[f'CAM_{b:02d}']
            if va['K'] != vb['K'] or va['T_cam_to_ego'] != vb['T_cam_to_ego']:
                raise ValueError(f'physical camera grouping changed: {method}')
        geometry.append([frame['T_ego_to_world'], [[vm[f'CAM_{c:02d}']['K'],vm[f'CAM_{c:02d}']['T_cam_to_ego']] for c in range(8)]])
    return hashlib.sha256(json.dumps(geometry, sort_keys=True).encode()).hexdigest()


def extract(args, method, rows):
    import torch
    from torch.utils.data import DataLoader, Subset
    from dwm.metrics.fvd import FrechetVideoDistance
    import dwm.metrics.fvd as fvd_module
    import pytorch_i3d
    manifest = args.root / METHODS[method] / 'stflow_manifest.jsonl'
    target = args.output / method
    target.mkdir(exist_ok=True)
    signature = {'manifest_sha256':sha_file(manifest), 'checkpoint_sha256':sha_file(args.checkpoint),
                 'script_sha256':sha_file(__file__), 'metric_sha256':sha_file(fvd_module.__file__),
                 'i3d_source_sha256':sha_file(pytorch_i3d.__file__), 'videos':len(rows), 'frames':19,
                 'batch_size':2,'torch':torch.__version__,'camera_names':[f'CAM_{i:02d}' for i in range(8)]}
    state = target/'signature.json'
    if state.exists() and json.loads(state.read_text()) != signature:
        raise ValueError('Feature cache signature changed; choose a new output directory')
    atomic_json(state,signature)
    dataset = VideoPairs(manifest, rows)
    missing = []
    for start in range(0,len(dataset),args.chunk_size):
        path = target/f'features_{start:06d}.npz'
        if not path.exists():
            missing.extend(range(start,min(start+args.chunk_size,len(dataset))))
    if not missing:
        print(f'[CACHED] {method}',flush=True)
        return
    metric = FrechetVideoDistance(str(args.checkpoint),sequence_count=19).cuda().eval()
    tap = FeatureTap()
    handle = metric.inception.register_forward_hook(tap)
    loader = DataLoader(Subset(dataset,missing),batch_size=2,num_workers=args.workers,
                        pin_memory=True,shuffle=False,prefetch_factor=2 if args.workers else None)
    collected_real,collected_fake,hash_real,hash_fake = [],[],[],[]
    cursor = 0
    verified = False
    for fake,real,fh,rh in loader:
        with torch.no_grad():
            metric.update(real.cuda(non_blocking=True), real=True)
            rf = tap.value
            metric.update(fake.cuda(non_blocking=True), real=False)
            ff = tap.value
        if not verified:
            np.testing.assert_allclose(rf.astype(np.float64).sum(0),metric.real_features_sum.cpu().numpy(),rtol=0,atol=1e-10)
            np.testing.assert_allclose(ff.astype(np.float64).T@ff,metric.fake_features_cov_sum.cpu().numpy(),rtol=1e-10,atol=1e-8)
            verified = True
            print('[CHECK] hook features reproduce original metric statistics',flush=True)
        collected_real.append(rf); collected_fake.append(ff)
        hash_real.extend(rh); hash_fake.extend(fh)
        cursor += len(fh)
        end = missing[cursor-1]+1
        if end % args.chunk_size == 0 or cursor == len(missing):
            start = end-len(hash_fake)
            path=target/f'features_{start:06d}.npz'
            tmp=Path(str(path)+'.tmp')
            with tmp.open('wb') as stream:
                np.savez(stream,real=np.concatenate(collected_real),fake=np.concatenate(collected_fake),
                         real_hash=np.array(hash_real),fake_hash=np.array(hash_fake))
                stream.flush();os.fsync(stream.fileno())
            os.replace(tmp,path)
            print(f'[FEATURES] {method} {end}/{len(dataset)}',flush=True)
            collected_real,collected_fake,hash_real,hash_fake=[],[],[],[]
    handle.remove()
    del metric
    torch.cuda.empty_cache()


def distance(real, fake, weights):
    import torch
    from dwm.metrics.fvd import _compute_fid
    mr,cr=moments(real,weights)
    mf,cf=moments(fake,weights)
    tensors=[torch.from_numpy(x) for x in (mr,cr,mf,cf)]
    return float(_compute_fid(*tensors))


def analyze(args):
    scores={}
    real_reference_hash=None
    choices=list(itertools.product((1,6),(2,5),(3,7)))
    variants={'A_8slot':np.ones(8),'B_5cam_00_04':np.array([1,1,1,1,1,0,0,0]),
              'C_5physical_balanced':np.array([1,.5,.5,.5,1,.5,.5,.5])}
    for choice in choices:
        weights=np.zeros(8);weights[[0,4,*choice]]=1
        variants['five_choice_'+'_'.join(map(str,choice))]=weights
    for camera in range(8):
        weights=np.zeros(8);weights[camera]=1
        variants[f'camera_{camera:02d}']=weights
    for method in METHODS:
        files=sorted((args.output/method).glob('features_*.npz'))
        parts=[dict(np.load(path)) for path in files]
        real=np.concatenate([v['real'] for v in parts]).reshape(args.videos,8,-1)
        fake=np.concatenate([v['fake'] for v in parts]).reshape(args.videos,8,-1)
        rh=np.concatenate([v['real_hash'] for v in parts]).reshape(args.videos,8)
        fh=np.concatenate([v['fake_hash'] for v in parts]).reshape(args.videos,8)
        if real_reference_hash is None:
            real_reference_hash=rh
        if not np.array_equal(real_reference_hash,rh):
            raise ValueError(f'{method}: real pixel data differs across methods')
        duplicates={f'{a}_{b}':{'real_exact_sequences':int((rh[:,a]==rh[:,b]).sum()),
                              'fake_exact_sequences':int((fh[:,a]==fh[:,b]).sum()),
                              'fake_feature_l2_mean':float(np.linalg.norm(fake[:,a]-fake[:,b],axis=-1).mean())} for a,b in PAIRS}
        values={name:distance(real,fake,weights) for name,weights in variants.items()}
        old=json.loads((args.root/METHODS[method]/'paired_fvd_result_all19.json').read_text())
        values['historical_8slot']=old['fvd']
        values['A_minus_historical']=values['A_8slot']-old['fvd']
        scores[method]={'fvd':values,'duplicate_generated_only_sequences':duplicates}
        atomic_json(args.output/'results.partial.json',scores)
        print('[RESULT]',method,json.dumps(values),flush=True)
    spread={}
    for name in variants:
        v=np.array([s['fvd'][name] for s in scores.values()])
        spread[name]={'variance_population':float(v.var()),'std_population':float(v.std()),'range':float(np.ptp(v)),'mean':float(v.mean())}
    shared4 = ['plucker','implicit','petr_correct','box']
    shared_spread = {}
    for name in variants:
        v = np.array([scores[m]['fvd'][name] for m in shared4])
        shared_spread[name] = {'variance_population':float(v.var()),'std_population':float(v.std()),'range':float(np.ptp(v))}
    nuplan_new = {'plucker':55.1495246887207,'implicit':63.078861236572266,'petr_correct':64.31684112548828,'box':59.71173858642578}
    nv = np.array(list(nuplan_new.values()))
    result={'common_four_methods':shared4, 'common_four_waymo_spread':shared_spread,
            'nuplan_reference_new_plucker':{'values':nuplan_new,'std_population':float(nv.std()),'variance_population':float(nv.var()),'range':float(np.ptp(nv)),'note':'Saved results, not recomputed here. NuPlan pvonly FVD unavailable.'},
            'methods_excluding_nocondition':list(METHODS),'videos_per_method':args.videos,
            'frames':19,'weighted_covariance':'frequency-mass: centered weighted sum / (sum weights - 1)',
            'warning':'Cross-method spread is descriptive, not repeated-run estimator variance. Paired bootstrap not computed.',
            'methods':scores,'spread':spread}
    atomic_json(args.output/'results.json',result)
    lines=['# Waymo FVD slot sensitivity (nocondition excluded)','',
           '| Method | Historical 8 | A: 8 slots | B: 5 selected | C: 5 physical balanced |',
           '|---|---:|---:|---:|---:|']
    for method,s in scores.items():
        v=s['fvd'];lines.append(f"| {method} | {v['historical_8slot']:.6f} | {v['A_8slot']:.6f} | {v['B_5cam_00_04']:.6f} | {v['C_5physical_balanced']:.6f} |")
    lines.extend(['','| Spread | A | B | C |','|---|---:|---:|---:|'])
    for key in ['variance_population','std_population','range']:
        lines.append('| '+key+' | '+' | '.join(f'{spread[name][key]:.6f}' for name in ['A_8slot','B_5cam_00_04','C_5physical_balanced'])+' |')
    lines.extend(['', '## Common four: plucker / implicit / corrected PETR / box', '',
                  '| Spread | A | B | C |', '|---|---:|---:|---:|'])
    for key in ['variance_population','std_population','range']:
        lines.append('| '+key+' | '+' | '.join(f'{shared_spread[name][key]:.6f}' for name in ['A_8slot','B_5cam_00_04','C_5physical_balanced'])+' |')
    lines.append(f'NuPlan NEW plucker reference, same four labels: std={nv.std():.6f}, variance={nv.var():.6f}, range={np.ptp(nv):.6f}.')
    lines.extend(['','See results.json for eight five-camera selections, per-slot FVD, and exact generated-sequence duplication counts.',
                  'A must reproduce historical FVD before attributing changes to camera weighting. Balanced covariance uses total mass 5000; A uses 8000.',
                  'Different generated images for identical calibration mean duplicate physical views, not necessarily duplicate generated samples.'])
    (args.output/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('[DONE]',args.output/'REPORT.md',flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--videos',type=int,default=1000)
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--chunk-size',type=int,default=256)
    parser.add_argument('--check-only',action='store_true')
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    preflight(args.root,args.output,args.videos)
    if args.check_only:
        return
    import torch
    torch.set_num_threads(4)
    np.random.seed(20260927);torch.manual_seed(20260927)
    torch.backends.cudnn.benchmark=False
    for method in METHODS:
        manifest = args.root / METHODS[method] / 'stflow_manifest.jsonl'
        with manifest.open() as stream:
            rows = [json.loads(line) for line in itertools.islice((s for s in stream if s.strip()), args.videos)]
        extract(args,method,rows)
        del rows
    analyze(args)


if __name__=='__main__':
    main()
