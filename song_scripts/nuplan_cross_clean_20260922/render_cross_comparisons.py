"""Offline cross-LoFTR qualitative comparisons; original STFlow metric is untouched."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from kornia.feature import LoFTR


class CrossOnlyEvaluator:
    """Binds existing evaluator methods without initializing unrelated RAFT."""
    def __init__(self, metric_class, checkpoint):
        self.device = torch.device('cpu')
        self.loftr_confidence = 0.1
        self.max_matches = 256
        self.min_matches = 16
        self.cross_gate_px = 16.0
        self.loftr_model = LoFTR(pretrained=None).eval()
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        self.loftr_model.load_state_dict(state.get('state_dict', state), strict=True)
        for name in ['load_rgb_image', 'load_valid_mask', 'load_frame_data', 'run_loftr',
                     'filter_matched_points', 'points_inside_image', 'points_inside_mask',
                     'fundamental_from_camera_to_ego', 'skew_matrix', 'sampson_error_px']:
            setattr(self, name, getattr(metric_class, name).__get__(self))


def color(error):
    fraction = min(max(float(error) / 32.0, 0.0), 1.0)
    return (int(45 + 200 * fraction), int(210 - 145 * fraction), 65)


def line_segment(line, width, height):
    a, b, c = line
    points = []
    if abs(b) > 1e-10:
        for x in [0, width - 1]:
            y = -(a*x+c)/b
            if 0 <= y < height: points.append((x, y))
    if abs(a) > 1e-10:
        for y in [0, height - 1]:
            x = -(b*y+c)/a
            if 0 <= x < width: points.append((x, y))
    return points[:2]


def draw_row(record, title, debug=False):
    images = [Image.open(p).convert('RGB') for p in record['images']]
    width, height = images[0].size
    assert images[1].size == (width, height)
    top = 72
    canvas = Image.new('RGB', (width*2, height+top+40), (20, 25, 31))
    canvas.paste(images[0], (0, top)); canvas.paste(images[1], (width, top))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 16)
    small = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 13)
    draw.text((12, 7), title, font=font, fill='white')
    draw.text((12, 31), f"Raw-Epi median: {record['raw_epi']:.2f}px   matches: {record['count']}   within 16px: {100*record['inlier_ratio']:.1f}%", font=small, fill='white')
    draw.text((12, 51), record['pair'][0], font=small, fill='#b8c3cc')
    draw.text((width+12, 51), record['pair'][1], font=small, fill='#b8c3cc')
    p0, p1, errors = map(np.asarray, [record['points0'], record['points1'], record['errors']])
    # Deterministic rank-spaced selection, not only low-error matches.
    indices = np.linspace(0, len(errors)-1, min(6 if debug else 40, len(errors))).astype(int)
    F = np.asarray(record['F'])
    for number, idx in enumerate(indices):
        x0, y0 = p0[idx]; x1, y1 = p1[idx]; col = color(errors[idx])
        if debug:
            line = F @ np.array([x0, y0, 1.0])
            ends = line_segment(line, width, height)
            if len(ends) == 2:
                draw.line([(x+width, y+top) for x, y in ends], fill=col, width=2)
            a,b,c = line; denominator = a*a+b*b
            if denominator > 1e-10:
                delta = (a*x1+b*y1+c)/denominator
                foot = (x1-a*delta, y1-b*delta)
                if 0 <= foot[0] < width and 0 <= foot[1] < height:
                    draw.line((x1+width,y1+top,foot[0]+width,foot[1]+top),fill='white',width=2)
            for x,y in [(x0,y0),(x1+width,y1)]:
                draw.text((x+5,y+top-16), str(number+1), font=small, fill='white',stroke_width=2,stroke_fill='black')
        else:
            draw.line((x0,y0+top,x1+width,y1+top),fill=col,width=1)
        for x,y in [(x0,y0),(x1+width,y1)]:
            draw.ellipse((x-3,y+top-3,x+3,y+top+3),fill=col,outline='black')
    note = '6 sampled matches | line: calibration F x | white: point-to-line offset (not Sampson distance)' if debug else 'Up to 40 sampled matches | fixed error colors: green 0px -> orange 16px -> red >=32px'
    draw.text((12,top+height+10),note,font=small,fill='#d3dbe1')
    return canvas


def evaluate(evaluator, item, root, frame_index, pair):
    frame = item['frames'][frame_index]
    views = [next(v for v in frame['views'] if v['camera']==camera) for camera in pair]
    data = evaluator.load_frame_data({'frames':[dict(frame,views=views)]}, str(root))
    p0,p1,_ = evaluator.run_loftr(*data['images'][0])
    p0,p1 = evaluator.filter_matched_points(p0,p1,*data['masks'][0])
    F = evaluator.fundamental_from_camera_to_ego(data['intrinsics'][0][0],data['transforms'][0][0],data['intrinsics'][0][1],data['transforms'][0][1])
    errors = evaluator.sampson_error_px(p0,p1,F)
    return dict(images=[str(root/v['image_path']) for v in views], pair=pair, count=len(errors),
                raw_epi=float(errors.median()) if len(errors) else None,
                inlier_ratio=float((errors<=16).float().mean()) if len(errors) else 0,
                points0=p0.tolist(),points1=p1.tolist(),errors=errors.tolist(),F=F.tolist())


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--repo',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--count',type=int,default=24)
    parser.add_argument('--threads',type=int,default=8)
    parser.add_argument('--max-low-error',type=float,default=None)
    parser.add_argument('--min-high-error',type=float,default=None)
    args=parser.parse_args(); torch.set_num_threads(args.threads)
    sys.path.insert(0,str(args.repo/'OpenDWM/src'))
    from dwm.metrics.stflow import STFlowEvaluator
    args.output.mkdir(parents=True,exist_ok=True)
    root=args.repo/'lyh_output/eval/waymo'
    roots={}; results={}
    for method in ['implicit','plucker','pvonly','box','nocondition']:
        paths=list(root.glob(method+'*merged1000/stflow_traj_result_gate16.json'))
        assert len(paths)==1, (method,paths)
        roots[method]=paths[0].parent
        obj=json.loads(paths[0].read_text()); conf=obj['eval_config']
        assert (conf['min_matches'],conf['max_matches'],conf['loftr_confidence'],conf['cross_gate_px'])==(16,256,0.1,16)
        results[method]={v['video_id']:v for v in obj['videos']}
    candidates=[]
    for vid in sorted(set.intersection(*(set(v) for v in results.values()))):
        pairs=set.intersection(*(set(v[vid]['pair_stats']) for v in results.values()))
        for pair in pairs:
            values=sorted((results[m][vid]['pair_stats'][pair]['cross_raw_epi_px'],m) for m in results)
            if not all(np.isfinite(v[0]) for v in values):continue
            if args.max_low_error is not None and values[0][0]>args.max_low_error:continue
            if args.min_high_error is not None and values[-1][0]<args.min_high_error:continue
            candidates.append(dict(video_id=vid,pair=pair.split('__'),low=values[0][1],high=values[-1][1],aggregate_gap=values[-1][0]-values[0][0]))
    candidates.sort(key=lambda x:x['aggregate_gap'],reverse=True)
    chosen=[];seen=set();pair_counts={}
    for c in candidates:
        key='__'.join(c['pair'])
        if c['video_id'] in seen or pair_counts.get(key,0)>=20:continue
        chosen.append(c);seen.add(c['video_id']);pair_counts[key]=pair_counts.get(key,0)+1
        if len(chosen)>=80:break
    manifests={}
    for method,p in roots.items():
        selected={c['video_id'] for c in chosen if method in [c['low'],c['high']]}
        manifests[method]={}
        for line in (p/'stflow_manifest.jsonl').open():
            # IDs are near the start, avoiding parsing most large manifest rows.
            if not any('"'+vid+'"' in line[:200] for vid in selected):continue
            item=json.loads(line)
            if item['video_id'] in selected:manifests[method][item['video_id']]=item
    evaluator=CrossOnlyEvaluator(STFlowEvaluator,args.weights)
    completed=[]
    for index,c in enumerate(chosen):
        t=[4,8,12,16][index%4]
        a,b=[manifests[m][c['video_id']] for m in [c['low'],c['high']]]
        assert a['reference_frame_count']==b['reference_frame_count']==3
        for fa,fb in zip(a['frames'],b['frames']):
            assert fa['frame_index']==fb['frame_index']
            assert fa.get('T_ego_to_world')==fb.get('T_ego_to_world')
            for va,vb in zip(fa['views'],fb['views']):
                assert va['camera']==vb['camera'] and va['K']==vb['K'] and va['T_cam_to_ego']==vb['T_cam_to_ego']
        for camera in c['pair']:
            va=next(v for v in a['frames'][t]['views'] if v['camera']==camera)
            vb=next(v for v in b['frames'][t]['views'] if v['camera']==camera)
            assert hashlib.sha256((roots[c['low']]/va['real_image_path']).read_bytes()).digest()==hashlib.sha256((roots[c['high']]/vb['real_image_path']).read_bytes()).digest()
        records=[]
        for method,item in [(c['low'],a),(c['high'],b)]:
            r=evaluate(evaluator,item,roots[method],t,c['pair']);r['method']=method;records.append(r)
        if min(r['count'] for r in records)<16:continue
        records.sort(key=lambda r:r['raw_epi'])
        gap=records[1]['raw_epi']-records[0]['raw_epi']
        print('CANDIDATE',index,c['video_id'],t,[(r['method'],round(r['raw_epi'],2),r['count']) for r in records],flush=True)
        if gap<4 or records[0]['inlier_ratio']<=records[1]['inlier_ratio']:continue
        if args.max_low_error is not None and records[0]['raw_epi']>args.max_low_error:continue
        if args.min_high_error is not None and records[1]['raw_epi']<args.min_high_error:continue
        number=len(completed)+1;prefix=f"{number:02d}_{c['video_id']}_t{t:02d}"
        report=dict(c,time_index=t,records=records,frame_gap=gap,alignment='all frame calibration/ego verified; selected paired-real images SHA256 identical')
        for debug,suffix in [(False,'matches'),(True,'epilines')]:
            rows=[draw_row(r,f"{'LOWER' if j==0 else 'HIGHER'} ERROR | {r['method']} | {c['video_id']} | t{t}",debug) for j,r in enumerate(records)]
            canvas=Image.new('RGB',(rows[0].width,rows[0].height*2+8),(10,13,17))
            canvas.paste(rows[0],(0,0));canvas.paste(rows[1],(0,rows[0].height+8))
            canvas.save(args.output/(prefix+'_'+suffix+'.jpg'),quality=95)
        (args.output/(prefix+'.json')).write_text(json.dumps(report,indent=2))
        completed.append(dict(prefix=prefix,video_id=c['video_id'],time_index=t,pair=c['pair'],lower=records[0]['method'],higher=records[1]['method'],low_error=records[0]['raw_epi'],high_error=records[1]['raw_epi']))
        (args.output/'index.json').write_text(json.dumps(completed,indent=2))
        print('SAVED',number,prefix,flush=True)
        if len(completed)>=args.count:break
    assert len(completed)>=args.count, len(completed)
    print('DONE',len(completed),flush=True)


if __name__=='__main__':main()
