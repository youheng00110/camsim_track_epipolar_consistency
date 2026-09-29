"""Sparse nuPlan LoFTR comparisons, aligned by full pose/calibration sequences."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from render_cross_comparisons import CrossOnlyEvaluator, evaluate, line_segment


def manifest_index(path):
    indexed={}; duplicates=[]
    with path.open() as f:
        while True:
            offset=f.tell();line=f.readline()
            if not line:break
            item=json.loads(line)
            geometry=[{'ego':frame.get('T_ego_to_world'),'index':frame['frame_index'],
                       'views':[(v['camera'],v['K'],v['T_cam_to_ego']) for v in frame['views']]} for frame in item['frames']]
            key=hashlib.sha256(json.dumps(geometry,separators=(',',':')).encode()).hexdigest()
            if key in indexed:duplicates.append(item['video_id']);continue
            indexed[key]={'video_id':item['video_id'],'offset':offset}
    return indexed,duplicates


def read_item(path, offset):
    with path.open() as f:f.seek(offset);return json.loads(f.readline())


def clean_image(records, sample, timestep, debug=False):
    width,height=Image.open(records[0]['images'][0]).size
    gap=12;head=34;label=30;rowheight=height+label
    canvas=Image.new('RGB',(width*2+gap,head+rowheight*2+10+30),(19,24,29))
    draw=ImageDraw.Draw(canvas)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',13)
    draw.text((10,8),f"nuPlan / {sample} / t{timestep:02d} / {' - '.join(records[0]['pair'])}",font=small,fill='#c5cdd3')
    aliases={'pvtrack2':'PVTrack2','pvbev':'PVBEV','bev_pv_epipolar32000':'BEV+PV+Epipolar'}
    for row,record in enumerate(records):
        ybase=head+row*(rowheight+10)
        draw.text((10,ybase+5),f"{aliases[record['method']]}    Cross Raw-Epi: {record['raw_epi']:.2f} px",font=font,fill='white')
        yimage=ybase+label
        for column,p in enumerate(record['images']):
            im=Image.open(p).convert('RGB');assert im.size==(width,height)
            canvas.paste(im,(column*(width+gap),yimage))
        points0=np.asarray(record['points0']);points1=np.asarray(record['points1']);errors=np.asarray(record['errors'])
        indices=np.linspace(0,len(errors)-1,min(3 if debug else 8,len(errors))).astype(int)
        for n,idx in enumerate(indices):
            x0,y0=points0[idx];x1,y1=points1[idx]
            fraction=min(errors[idx]/16,1)
            col=(int(60+180*fraction),int(205-100*fraction),int(145-60*fraction))
            offset=width+gap
            if debug:
                line=np.asarray(record['F'])@np.array([x0,y0,1])
                ends=line_segment(line,width,height)
                if len(ends)==2:draw.line([(x+offset,y+yimage) for x,y in ends],fill=col,width=1)
                a,b,c=line;denominator=a*a+b*b
                if denominator>1e-10:
                    d=(a*x1+b*y1+c)/denominator;fx,fy=x1-a*d,y1-b*d
                    if 0<=fx<width and 0<=fy<height:draw.line((x1+offset,y1+yimage,fx+offset,fy+yimage),fill='white',width=1)
                for x,y in [(x0,y0),(x1+offset,y1)]:draw.text((x+5,y+yimage-13),str(n+1),font=small,fill='white',stroke_width=1,stroke_fill='black')
            else:draw.line((x0,y0+yimage,x1+offset,y1+yimage),fill=col,width=1)
            for x,y in [(x0,y0),(x1+offset,y1)]:draw.ellipse((x-2,y+yimage-2,x+2,y+yimage+2),fill=col)
        record['display_indices_epilines' if debug else 'display_indices_matches']=indices.tolist()
    note='3 matches / calibration epilines / white: perpendicular offset' if debug else '8 matches shown / score uses all valid matches / fixed colors: green 0 px - orange >=16 px'
    draw.text((10,canvas.height-21),note,font=small,fill='#b3bdc4')
    return canvas


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--repo',type=Path,required=True);parser.add_argument('--weights',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);parser.add_argument('--count',type=int,default=24)
    args=parser.parse_args();torch.set_num_threads(8);sys.path.insert(0,str(args.repo/'OpenDWM/src'))
    from dwm.metrics.stflow import STFlowEvaluator
    root=args.repo/'lyh_output/eval/nuplanhard1000';roots={m:root/(m+'_merged1000') for m in ['pvtrack2','pvbev','bev_pv_epipolar32000']}
    indexes={};results={};duplicates={}
    for method,path in roots.items():
        indexes[method],duplicates[method]=manifest_index(path/'stflow_manifest.jsonl')
        d=json.loads((path/'stflow_traj_result_gate16.json').read_text())
        conf=d['eval_config'];assert (conf['min_matches'],conf['max_matches'],conf['loftr_confidence'],conf['cross_gate_px'])==(16,256,0.1,16)
        results[method]={v['video_id']:v for v in d['videos']}
        print('INDEX',method,len(indexes[method]),'duplicates',duplicates[method],flush=True)
    common=set.intersection(*(set(v) for v in indexes.values()));candidates=[]
    for key in sorted(common):
        videos={m:results[m][indexes[m][key]['video_id']] for m in roots}
        pairs=set.intersection(*(set(v['pair_stats']) for v in videos.values()))
        for pair in pairs:
            values=sorted((videos[m]['pair_stats'][pair]['cross_raw_epi_px'],m) for m in roots)
            if not all(np.isfinite(v[0]) for v in values):continue
            delta=values[-1][0]-values[0][0]
            if 2<=delta<=10 and values[0][0]<=6 and values[-1][0]<=16:
                candidates.append(dict(key=key,pair=pair.split('__'),low=values[0][1],high=values[-1][1],aggregate_gap=delta))
    candidates.sort(key=lambda c:abs(c['aggregate_gap']-5))
    seen=set();chosen=[]
    for c in candidates:
        if c['key'] in seen:continue
        chosen.append(c);seen.add(c['key'])
        if len(chosen)>=160:break
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'alignment.json').write_text(json.dumps(dict(unique={m:len(v) for m,v in indexes.items()},duplicates=duplicates,common=len(common),candidates=len(chosen)),indent=2))
    print('CANDIDATES',len(chosen),flush=True)
    evaluator=CrossOnlyEvaluator(STFlowEvaluator,args.weights);completed=[]
    for idx,c in enumerate(chosen):
        t=[4,8,12,16][idx%4]
        methods=[c['low'],c['high']]
        items={m:read_item(roots[m]/'stflow_manifest.jsonl',indexes[m][c['key']]['offset']) for m in methods}
        assert all(item['reference_frame_count']==3 for item in items.values())
        same_real=True
        for camera in c['pair']:
            digests=[]
            for m in methods:
                v=next(v for v in items[m]['frames'][t]['views'] if v['camera']==camera)
                digests.append(hashlib.sha256((roots[m]/v['real_image_path']).read_bytes()).hexdigest())
            same_real=same_real and len(set(digests))==1
        if not same_real:print('SKIP real image mismatch',idx,flush=True);continue
        records=[]
        for m in methods:
            record=evaluate(evaluator,items[m],roots[m],t,c['pair']);record.update(method=m,video_id=items[m]['video_id']);records.append(record)
        if min(r['count'] for r in records)<16:continue
        records.sort(key=lambda r:r['raw_epi']);delta=records[1]['raw_epi']-records[0]['raw_epi']
        print('CHECK',idx,t,[(r['method'],round(r['raw_epi'],2)) for r in records],flush=True)
        if not (2<=delta<=10 and records[0]['raw_epi']<=8 and records[1]['raw_epi']<=18):continue
        sample=indexes['bev_pv_epipolar32000'][c['key']]['video_id'];prefix=f"{len(completed)+1:02d}_{sample}_t{t:02d}"
        for debug,suffix in [(False,'matches'),(True,'epilines')]:clean_image(records,sample,t,debug).save(args.output/(prefix+'_'+suffix+'.jpg'),quality=95)
        report=dict(c,time_index=t,records=records,alignment='Full pose/calibration sequence hash; deduplicated per method; selected paired-real SHA256 identical')
        (args.output/(prefix+'.json')).write_text(json.dumps(report,indent=2))
        completed.append(dict(prefix=prefix,video_id=sample,time_index=t,pair=c['pair'],lower=records[0]['method'],higher=records[1]['method'],low_error=records[0]['raw_epi'],high_error=records[1]['raw_epi']))
        (args.output/'index.json').write_text(json.dumps(completed,indent=2));print('SAVED',len(completed),prefix,flush=True)
        if len(completed)>=args.count:break
    assert len(completed)==args.count,len(completed)
    print('DONE',len(completed),flush=True)


if __name__=='__main__':main()
