"""Evaluate each dataset separately with dataset-specific classes."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoProcessor
from peft import PeftModel
from v1_core import MODEL_ID, SegDataset, V1Segmenter, read_maps, single_collate, to_device


def restore_model(maps, checkpoint, device):
    model = V1Segmenter(maps)
    # Remove freshly initialized LoRA, load checkpoint adapter without touching original base.
    model.mllm = PeftModel.from_pretrained(model.mllm.unload(), str(Path(checkpoint)/"lora_weights"), is_trainable=False)
    state = torch.load(Path(checkpoint)/"v1_heads.pth", map_location="cpu", weights_only=True)
    for name in ("spatial_proj", "text_proj", "shared_decoder", "heads"):
        getattr(model, name).load_state_dict(state[name])
        getattr(model, name).to(device)
    return model.eval()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", required=True, choices=["LoveDA","Potsdam","Vaihingen"])
    p.add_argument("--val-json", default="datasets/val_dataset.json")
    p.add_argument("--maps", default="configs/label_maps.json")
    p.add_argument("--limit", type=int, default=32, help="0 = all validation samples")
    p.add_argument("--visualize", type=int, default=6)
    p.add_argument("--output", default="results/v1_eval")
    args=p.parse_args()
    maps=read_maps(args.maps)
    ds=SegDataset(args.val_json, AutoProcessor.from_pretrained(MODEL_ID), maps, [args.dataset])
    if args.limit: ds=Subset(ds, range(min(args.limit,len(ds))))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type!="cuda": raise RuntimeError("V1 4-bit inference requires CUDA GPU")
    model=restore_model(maps,args.checkpoint,device)
    loader=DataLoader(ds,batch_size=1,num_workers=0,collate_fn=single_collate)
    n=maps[args.dataset]["num_classes"]
    conf=np.zeros((n,n),dtype=np.int64)
    out=Path(args.output)/args.dataset
    out.mkdir(parents=True,exist_ok=True)
    palette=np.array([(255,255,255),(220,60,60),(230,200,45),(40,110,220),
                      (175,110,185),(45,180,95),(70,200,200)],dtype=np.uint8)
    for i,batch in enumerate(loader):
        with torch.no_grad(), torch.autocast("cuda",dtype=torch.bfloat16):
            pred=model(to_device(batch,device)).argmax(1)[0].cpu().numpy()
        gt=batch["target_mask"][0].numpy()
        valid=gt!=255
        inds=n*gt[valid].astype(np.int64)+pred[valid].astype(np.int64)
        conf+=np.bincount(inds,minlength=n*n).reshape(n,n)
        if i<args.visualize:
            orig_path=(ds.dataset.data[ds.indices[i]]["image"] if isinstance(ds,Subset) else ds.data[i]["image"])
            orig=Image.open(orig_path).convert("RGB").resize((512,512))
            gt_rgb=palette[np.clip(gt,0,6)].copy(); gt_rgb[~valid]=0
            pd_rgb=palette[pred].copy(); pd_rgb[~valid]=0
            side=Image.new("RGB",(1536,512)); side.paste(orig,(0,0));
            side.paste(Image.fromarray(gt_rgb),(512,0));side.paste(Image.fromarray(pd_rgb),(1024,0))
            side.save(out/f"compare_{i:03d}.png")
    tp=np.diag(conf).astype(float)
    union=conf.sum(0)+conf.sum(1)-tp
    iou=np.divide(tp,union,out=np.full(n,np.nan),where=union>0)*100
    oa=tp.sum()/max(1,conf.sum())*100
    report={"checkpoint":args.checkpoint,"dataset":args.dataset,"samples":len(ds),
            "OA":round(float(oa),4),"mIoU_present":round(float(np.nanmean(iou)),4) if np.isfinite(iou).any() else None,
            "per_class_iou":{k: (round(float(v),4) if np.isfinite(v) else None)
                             for k,v in zip(maps[args.dataset]["class_names"],iou)},
            "confusion_matrix":conf.tolist()}
    (out/"metrics.json").write_text(json.dumps(report,indent=2,ensure_ascii=False))
    with (out/"per_class_iou.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["class","IoU_pct"])
        for k,v in report["per_class_iou"].items(): w.writerow([k,v])
    print(json.dumps(report,indent=2,ensure_ascii=False))


if __name__=="__main__": main()
