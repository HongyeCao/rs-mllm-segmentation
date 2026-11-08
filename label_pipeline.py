#!/usr/bin/env python3
"""Audit and non-destructively remap semantic masks for LoveDA/Potsdam/Vaihingen.
No inference about local ISPRS numeric IDs is made. Python >=3.9, numpy, pillow.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image

DATASETS = ('LoveDA', 'Potsdam', 'Vaihingen')

def get_dataset(rec):
    ds = rec.get('dataset')
    if ds in DATASETS:
        return ds
    tag = rec.get('id', '').split('_', 1)[0]
    if tag in DATASETS:
        return tag
    parts = Path(rec['mask']).parts
    for name in DATASETS:
        if name in parts:
            return name
    raise ValueError(f"Cannot determine dataset: {rec.get('id')} | {rec.get('mask')}")

def load_json(path):
    with open(path, encoding='utf8') as f:
        return json.load(f)

def mask_array(path):
    with Image.open(path) as im:
        # IMPORTANT: preserve actual indexed values, never use convert('L') here.
        if im.mode not in ('L', 'P', 'I', 'I;16'):
            raise ValueError(f"Non-indexed mask {path} (mode={im.mode}); inspect original RGB color-code converter")
        arr = np.asarray(im)
    if arr.ndim != 2:
        raise ValueError(f"Not a 2-D indexed mask: {path} shape={arr.shape}")
    return arr

def audit(paths, limit=None, out='label_audit.json'):
    stats = {d: {'samples':0,'modes':Counter(),'pixels':Counter(),'examples':[]} for d in DATASETS}
    for file in paths:
        for rec in load_json(file):
            ds = get_dataset(rec)
            if limit is not None and stats[ds]['samples'] >= limit:
                continue
            p = Path(rec['mask'])
            if not p.is_file():
                raise FileNotFoundError(f"Run audit on the WSL machine containing masks: {p}")
            with Image.open(p) as img:
                stats[ds]['modes'][img.mode] += 1
            arr=mask_array(p)
            vals, cnts=np.unique(arr, return_counts=True)
            stats[ds]['pixels'].update({int(x):int(n) for x,n in zip(vals,cnts)})
            stats[ds]['samples'] += 1
            if len(stats[ds]['examples']) < 3:
                stats[ds]['examples'].append(str(p))
    result={}
    for ds,s in stats.items():
        tot=sum(s['pixels'].values())
        result[ds]={'samples':s['samples'], 'modes':dict(s['modes']),
                    'values':{str(k):{'pixels':v,'pct':round(v/tot*100,4) if tot else 0}
                              for k,v in sorted(s['pixels'].items())},
                    'examples':s['examples']}
        print(f"\n{ds}: {s['samples']} masks, modes={dict(s['modes'])}")
        print('  pixel distribution:',[(k, round(v/tot*100,2)) for k,v in sorted(s['pixels'].items())] if tot else [])
    with open(out,'w',encoding='utf8') as f:
        json.dump(result,f,indent=2,ensure_ascii=False)
    print('Saved:',out)

def build(paths, map_path, output_root, json_output_dir):
    cfg=json.load(open(map_path,encoding='utf8'))
    root=Path(output_root)
    outdir=Path(json_output_dir)
    outdir.mkdir(parents=True,exist_ok=True)
    for file in paths:
        source=load_json(file)
        output=[]
        for i, rec in enumerate(source):
            ds=get_dataset(rec)
            info=cfg.get(ds)
            if not info or not info.get('verified',False):
                raise ValueError(f"{ds}: mapping is NOT verified. Inspect mask decoding code and set verified=true in config only after checking.")
            mapping={int(k):int(v) for k,v in info['raw_to_train'].items()}
            nclasses=int(info['num_classes'])
            if sorted(set(v for v in mapping.values() if v!=255)) != list(range(nclasses)):
                raise ValueError(f"{ds}: mapped classes must be 0..{nclasses-1}")
            arr=mask_array(rec['mask'])
            found=set(int(z) for z in np.unique(arr))
            if not found.issubset(set(mapping)):
                raise ValueError(f"{ds} sample {rec['mask']} has unknown raw IDs: {sorted(found-set(mapping))}")
            outmask=np.full(arr.shape, 255,dtype=np.uint8)
            for raw,dest in mapping.items():
                outmask[arr==raw]=dest
            if not np.any(outmask!=255):
                # Retain samples only if there is at least one supervised pixel.
                continue
            # Avoid flattening paths with same basename. Preserve split+source dataset+parent image stem.
            original=Path(rec['mask'])
            split='train' if 'train' in Path(file).stem.lower() else 'val'
            newmask=root/ds/split/'ann'/f"{original.stem}.png"
            if newmask.exists():
                # This is safe only if mapping produces same data. Prevent silent collisions.
                old=np.asarray(Image.open(newmask))
                if not np.array_equal(old,outmask):
                    raise RuntimeError(f"Collision with different pixel labels: {newmask}")
            else:
                newmask.parent.mkdir(parents=True,exist_ok=True)
                Image.fromarray(outmask,mode='L').save(newmask)
            newrec=dict(rec)
            newrec['dataset']=ds
            newrec['mask']=str(newmask.resolve())
            newrec['num_classes']=nclasses
            output.append(newrec)
            if (i+1)%5000==0:
                print(f"{Path(file).name}: {i+1}/{len(source)}")
        outpath=outdir/(Path(file).stem+'_routed.json')
        with open(outpath,'w',encoding='utf8') as f:
            json.dump(output,f,ensure_ascii=False,indent=2)
        print('Wrote',outpath,'samples',len(output))

def main():
    p=argparse.ArgumentParser()
    sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('audit',help='Scan original mask pixel IDs without modifying files')
    a.add_argument('--json',nargs='+',default=['datasets/train_dataset.json','datasets/val_dataset.json'])
    a.add_argument('--limit',type=int,default=200,help='Per-dataset sample limit; use 0 for all masks')
    a.add_argument('--out',default='label_audit.json')
    b=sub.add_parser('build',help='Make separate remapped masks and routed JSONs after verifying maps')
    b.add_argument('--json',nargs='+',default=['datasets/train_dataset.json','datasets/val_dataset.json'])
    b.add_argument('--maps',default='label_maps.json')
    b.add_argument('--out-masks',default='datasets/remapped_masks')
    b.add_argument('--out-json',default='datasets')
    args=p.parse_args()
    if args.mode=='audit':
        audit(args.json,None if args.limit==0 else args.limit,args.out)
    else:
        build(args.json,args.maps,args.out_masks,args.out_json)
if __name__=='__main__':
    main()
