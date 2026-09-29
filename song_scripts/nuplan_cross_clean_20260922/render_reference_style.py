"""Reference-style filtered correspondence illustrations; evaluation scores unchanged."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from render_cross_comparisons import CrossOnlyEvaluator


def sparse_selection(indices, points0, points1, confidence, limit):
    selected=[]
    for i in sorted(indices,key=lambda i:float(confidence[i]),reverse=True):
        if all(np.linalg.norm(points0[i]-points0[j])>=14 and np.linalg.norm(points1[i]-points1[j])>=14 for j in selected):selected.append(i)
        if len(selected)>=limit:break
    return selected


def render_panel(record, sample, time_index):
    images=[Image.open(p).convert('RGB') for p in record['images']]
    w,h=images[0].size;assert images[1].size==(w,h)
    canvas=Image.new('RGB',(w*2,h+102),'white');canvas.paste(images[0],(0,65));canvas.paste(images[1],(w,65))
    draw=ImageDraw.Draw(canvas)
    title=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',21)
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',14)
    small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',12)
    draw.text((8,3),'Cross-view Correspondence',font=title,fill='black')
    aliases={'pvtrack2':'PVTrack2','pvbev':'PVBEV','bev_pv_epipolar32000':'BEV+PV+Epipolar'}
    draw.text((8,29),f"{aliases[record['method']]}   {sample}   {' -> '.join(record['pair'])}   t={time_index}",font=font,fill='black')
    draw.text((8,47),f"Raw-Epi (all {record['count']} valid matches): {record['raw_epi']:.2f}px     shown: {len(record['shown'])}/{record['quality_count']} filtered matches",font=small,fill='#333333')
    for i in record['shown']:
        x0,y0=record['points0'][i];x1,y1=record['points1'][i]
        col=(0,182,53) if record['errors'][i]<=10 else (229,43,43)
        draw.line((x0,y0+65,x1+w,y1+65),fill=col,width=1)
        for x,y in [(x0,y0),(x1+w,y1)]:draw.ellipse((x-2,y+63,x+2,y+67),fill=col)
    if not record['shown']:draw.text((12,80),'No matches pass the display-quality filter',font=font,fill='white',stroke_width=1,stroke_fill='black')
    y=h+77;draw.rectangle((10,y,23,y+13),fill=(0,182,53))
    draw.text((31,y-1),'Display: conf>=0.70, reciprocal<=1.5px, epi<=10px',font=small,fill='black')
    draw.rectangle((590,y,603,y+13),fill=(229,43,43));draw.text((610,y-1),'10<epi<=12px (at most 1 shown)',font=small,fill='black')
    return canvas


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(8);sys.path.insert(0,str(a.repo/'OpenDWM/src'))
    from dwm.metrics.stflow import STFlowEvaluator
    e=CrossOnlyEvaluator(STFlowEvaluator,a.weights);a.output.mkdir(parents=True,exist_ok=True)
    index=json.loads((a.source/'index.json').read_text());finished=[]
    needed={}; manifests={}
    for case in index:
        report=json.loads((a.source/(case['prefix']+'.json')).read_text())
        for r in report['records']:needed.setdefault(r['method'],set()).add(r['video_id'])
    for method,ids in needed.items():
        root=a.repo/'lyh_output/eval/nuplanhard1000'/(method+'_merged1000');manifests[method]={}
        for line in (root/'stflow_manifest.jsonl').open():
            if any('"'+vid+'"' in line[:200] for vid in ids):
                item=json.loads(line)
                if item['video_id'] in ids:manifests[method][item['video_id']]=item

    for case in index:
        report=json.loads((a.source/(case['prefix']+'.json')).read_text())
        for record in report['records']:
            item=manifests[record['method']][record['video_id']];frame=item['frames'][case['time_index']]
            pair=['CAM_07','CAM_00'];views=[next(v for v in frame['views'] if v['camera']==c) for c in pair]
            root=a.repo/'lyh_output/eval/nuplanhard1000'/(record['method']+'_merged1000')
            data=e.load_frame_data({'frames':[dict(frame,views=views)]},str(root))
            image0,image1=data['images'][0]
            p0,p1,conf=e.run_loftr(image0,image1)
            keep=e.points_inside_mask(p0,data['masks'][0][0]) & e.points_inside_mask(p1,data['masks'][0][1])
            p0,p1,conf=p0[keep],p1[keep],conf[keep]
            F=e.fundamental_from_camera_to_ego(data['intrinsics'][0][0],data['transforms'][0][0],data['intrinsics'][0][1],data['transforms'][0][1])
            raw=e.sampson_error_px(p0,p1,F)
            assert len(raw)>=16
            record.update(images=[str(root/v['image_path']) for v in views],pair=pair,count=len(raw),raw_epi=float(raw.median()),inlier_ratio=float((raw<=16).float().mean()),errors=raw.tolist(),F=F.tolist(),points0=p0.tolist(),points1=p1.tolist())
            record['real_sha256']=[__import__('hashlib').sha256((root/v['real_image_path']).read_bytes()).hexdigest() for v in views]

            q1,q0,reverse_conf=e.run_loftr(image1,image0)
            keep=reverse_conf>=0.7;q0=q0[keep];q1=q1[keep]
            reciprocal=torch.full((len(p0),),float('inf'))
            if len(q0):reciprocal=torch.maximum(torch.cdist(p0,q0),torch.cdist(p1,q1)).min(dim=1).values
            errors=np.asarray(record['errors']);quality=(conf.numpy()>=0.7)&(reciprocal.numpy()<=1.5)&(errors<=12)
            green=np.flatnonzero(quality&(errors<=10)).tolist();red=np.flatnonzero(quality&(errors>10)).tolist()
            selected=sparse_selection(green,p0.numpy(),p1.numpy(),conf.numpy(),8)
            # Show only a small illustrative subset. This is not the inlier ratio.
            if len(selected)>=7 and red:
                selected=selected[:7]+sparse_selection(red,p0.numpy(),p1.numpy(),conf.numpy(),1)
            record.update(shown=selected,quality_count=int(quality.sum()),green_count=len(green),red_count=len(red),confidence=conf.tolist(),reciprocal_px=[float(x) if np.isfinite(x) else None for x in reciprocal],display_filter={'confidence':0.7,'reciprocal_px':1.5,'max_epi_px':12,'green_epi_px':10,'max_red_shown':1})
            render_panel(record,case['video_id'],case['time_index']).save(a.output/(case['prefix']+'_'+record['method']+'.png'))
        assert report['records'][0]['real_sha256']==report['records'][1]['real_sha256']
        report['records'].sort(key=lambda r:r['raw_epi'])
        report['pair']=['CAM_07','CAM_00'];case['pair']=report['pair']
        case.update(lower=report['records'][0]['method'],higher=report['records'][1]['method'],low_error=report['records'][0]['raw_epi'],high_error=report['records'][1]['raw_epi'])
        panels=[render_panel(r,case['video_id'],case['time_index']) for r in report['records']]
        combined=Image.new('RGB',(panels[0].width,panels[0].height*2+12),'white');combined.paste(panels[0],(0,0));combined.paste(panels[1],(0,panels[0].height+12))
        combined.save(a.output/(case['prefix']+'_comparison.jpg'),quality=95)
        (a.output/(case['prefix']+'.json')).write_text(json.dumps(report,indent=2))
        case['shown_counts']=[len(r['shown']) for r in report['records']];finished.append(case)
        (a.output/'index.json').write_text(json.dumps(finished,indent=2))
        print('SAVED',len(finished),case['prefix'],case['shown_counts'],flush=True)
    print('DONE',len(finished),flush=True)


if __name__=='__main__':main()
