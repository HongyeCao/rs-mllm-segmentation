import os
import json
import random

def build_mllm_json(split="train"):
    base_dir = os.path.expanduser("~/rs_project/datasets")
    datasets = ["LoveDA", "Vaihingen", "Potsdam"]
    
    # 为了提升模型的泛化能力，我们为不同的数据集准备多样化的 Instruction
    prompts = [
        "Please segment all the semantic targets in this remote sensing image.",
        "Can you identify and segment the specific regions in this aerial patch?",
        "Provide the pixel-level segmentation mask for this geographic area."
    ]
    
    all_samples = []
    
    for ds_name in datasets:
        img_dir = os.path.join(base_dir, ds_name, split, "img")
        ann_dir = os.path.join(base_dir, ds_name, split, "ann")
        
        if not os.path.exists(img_dir) or not os.path.exists(ann_dir):
            print(f"Warning: {ds_name} 的 {split} 目录不存在，已跳过。")
            continue
            
        img_files = [f for f in os.listdir(img_dir) if f.endswith(('.png', '.jpg', '.tif'))]
        
        for img_name in img_files:
            img_path = os.path.join(img_dir, img_name)
            # 假设掩码的名称与原图一致，或者后缀为 png
            ann_name = os.path.splitext(img_name)[0] + ".png"
            ann_path = os.path.join(ann_dir, ann_name)
            
            if os.path.exists(ann_path):
                sample = {
                    "id": f"{ds_name}_{split}_{os.path.splitext(img_name)[0]}",
                    "image": img_path,
                    "mask": ann_path,
                    "conversations": [
                        {"from": "human", "value": f"<image>\n{random.choice(prompts)}"},
                        {"from": "gpt", "value": "Sure, [SEG]."}
                    ]
                }
                all_samples.append(sample)
                
    # 打乱数据集顺序，避免模型在训练时连续看到同一个数据集导致梯度震荡
    random.shuffle(all_samples)
    
    out_json = os.path.join(base_dir, f"{split}_dataset.json")
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(all_samples, f, indent=4, ensure_ascii=False)
        
    print(f"[{split}] 数据集构建完成！共融合 {len(all_samples)} 张切片，保存至: {out_json}")

if __name__ == "__main__":
    build_mllm_json("train")
    build_mllm_json("val")
