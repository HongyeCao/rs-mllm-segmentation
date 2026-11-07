# ==============================================================================
# 脚本名称：build_multi_dataset_json.py
# 作用：自动化扫描遥感数据集目录，批量生成训练/验证所需的多模态标准 JSON 文件
# ==============================================================================

import os
import json
from glob import glob

def generate_dataset_json(images_dir, masks_dir, output_json_path):
    """
    扫描指定目录下的图像与掩码文件，组装成 Qwen2-VL 多模态微调所需的标准对话格式
    """
    image_extensions = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff")
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(images_dir, "**", ext), recursive=True))
    
    image_paths.sort()
    dataset_records = []
    
    print(f"🔍 正在扫描目录: {images_dir}，共找到 {len(image_paths)} 张影像文件...")
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        # 自动推导对应的掩码路径 (支持同名或规则匹配)
        base_name, _ = os.path.splitext(filename)
        mask_candidates = [
            os.path.join(masks_dir, filename),
            os.path.join(masks_dir, base_name + ".png"),
            os.path.join(masks_dir, base_name + ".tif")
        ]
        
        mask_path = None
        for cand in mask_candidates:
            if os.path.exists(cand):
                mask_path = cand
                break
                
        if mask_path is None:
            continue # 如果找不到对应的掩码则跳过
            
        # 构筑遥感领域的专家先验 Prompt 与对话对
        record = {
            "image": img_path,
            "mask": mask_path,
            "conversations": [
                {
                    "role": "user",
                    "value": "<image>\n请作为遥感智能解译专家，精确识别并勾勒出该高分遥感影像中的各类地物目标（含背景、建筑、道路、水体、植被等）的像素级空间边缘轮廓。"
                },
                {
                    "role": "assistant",
                    "value": "好的，我已经成功解析了该遥感影像的全局语义特征与局部多尺度空间结构，正在输出高精度的像素级语义分割预测掩码。"
                }
            ]
        }
        dataset_records.append(record)
        
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_records, f, ensure_ascii=False, indent=4)
        
    print(f"✅ 成功生成多模态训练标准 JSON 文件！有效样本数: {len(dataset_records)}，已保存至: {output_json_path}")

if __name__ == "__main__":
    # 示例用法：您可以根据本地实际的遥感数据集路径进行调整
    # 假设您的数据集存放在 datasets/train/img 和 datasets/train/mask
    train_img_dir = "datasets/train/img"
    train_mask_dir = "datasets/train/mask"
    output_json = "datasets/train_dataset.json"
    
    if os.path.exists(train_img_dir) and os.path.exists(train_mask_dir):
        generate_dataset_json(train_img_dir, train_mask_dir, output_json)
    else:
        print("💡 提示：未检测到默认的 datasets/train 目录，请修改脚本中的路径变量或直接运行您的自定义构建逻辑。")