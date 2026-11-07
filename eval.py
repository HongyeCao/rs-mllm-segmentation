import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel
import numpy as np
from PIL import Image

# 从您的 train.py 中直接复用数据集和模型定义，确保结构 100% 一致
from train import RSMultiModalDataset, RS_MLLM_Segmenter

# ==========================================
# 1. 定义遥感地物类别与标准配色表 (RGB)
# ==========================================
# 请根据您实际使用的数据集 (如 LoveDA / Vaihingen) 修改类别名称
CLASS_NAMES = ["Background/Clutter", "Building", "Road", "Water", "Barren/LowVeg", "Tree/Forest"]

# 为 6 个类别分配高对比度的科研标准配色，255 (忽略区域) 设为纯黑色
PALETTE = {
    0: (255, 255, 255), # 类别 0: 白色
    1: (255, 0, 0),     # 类别 1: 红色 (常用于建筑 Building)
    2: (255, 255, 0),   # 类别 2: 黄色 (常用于道路 Road)
    3: (0, 0, 255),     # 类别 3: 蓝色 (常用于水体 Water)
    4: (159, 129, 183), # 类别 4: 紫色 (常用于裸地/低矮植被)
    5: (0, 255, 0),     # 类别 5: 绿色 (常用于树木/森林)
    255: (0, 0, 0)      # 忽略的边缘区域: 纯黑色
}

def mask_to_color(mask_np):
    """将单通道类别索引图 (H, W) 转换为彩色 RGB 图像 (H, W, 3)"""
    h, w = mask_np.shape
    color_img = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        color_img[mask_np == class_id] = color
    return Image.fromarray(color_img)

def load_trained_model(checkpoint_dir, num_classes=6, device="cuda"):
    """安全加载分离保存的 4-bit 基础模型 + LoRA 权重 + 自定义分割解码头"""
    print(f"🔄 正在从 {checkpoint_dir} 加载训练好的模型权重...")
    
    # 1. 实例化基础结构
    model = RS_MLLM_Segmenter(num_classes=num_classes)
    
    # 2. 卸载初始化的随机 LoRA，挂载训练好的 LoRA 权重
    lora_path = os.path.join(checkpoint_dir, "lora_weights")
    base_mllm = model.mllm.unload()
    model.mllm = PeftModel.from_pretrained(base_mllm, lora_path)
    
    # 3. 加载自定义空间投影器与分割解码器权重
    heads_path = os.path.join(checkpoint_dir, "seg_heads.pth")
    custom_heads = torch.load(heads_path, map_location="cpu")
    model.spatial_proj.load_state_dict(custom_heads["spatial_proj"])
    model.text_proj.load_state_dict(custom_heads["text_proj"])
    model.mask_decoder.load_state_dict(custom_heads["mask_decoder"])
    
    model.to(device)
    model.eval()
    print("✅ 模型权重加载完毕，已切换至推理评估模式 (Eval Mode)！")
    return model

# ==========================================
# 2. 定量评估 (mIoU / OA) 与定性可视化主流程
# ==========================================
def evaluate(checkpoint_dir="checkpoints/step_2000", json_path="datasets/train_dataset.json", num_eval_samples=200, num_vis=10):
    NUM_CLASSES = 6
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    model = load_trained_model(checkpoint_dir, num_classes=NUM_CLASSES, device=device)
    
    # 如果您有独立的验证集 val_dataset.json，请将 json_path 替换为验证集路径
    dataset = RSMultiModalDataset(json_path, processor)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2)
    
    os.makedirs("eval_results", exist_ok=True)
    
    # 初始化混淆矩阵 (Confusion Matrix) 用于计算 mIoU
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.int64)
    vis_count = 0
    
    print(f"🚀 开始评估前 {num_eval_samples} 张样本，并将生成 {num_vis} 张可视化对比图...")
    
    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= num_eval_samples:
                break
                
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            mm_token_type_ids = batch["mm_token_type_ids"].to(device)
            targets = batch["target_mask"].to(device).long()
            
            # 过滤非法像素
            targets[(targets >= NUM_CLASSES) & (targets != 255)] = 255
            if (targets != 255).sum() == 0:
                continue
                
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
                preds = torch.argmax(logits, dim=1) # shape: (1, 512, 512)
                
            pred_np = preds.squeeze(0).cpu().numpy()
            target_np = targets.squeeze(0).cpu().numpy()
            
            # 1. 更新混淆矩阵（忽略值为 255 的像素）
            valid_mask = (target_np != 255)
            valid_preds = pred_np[valid_mask]
            valid_targets = target_np[valid_mask]
            
            # 利用 bincount 快速统计二维混淆矩阵
            indices = NUM_CLASSES * valid_targets + valid_preds
            counts = np.bincount(indices, minlength=NUM_CLASSES**2)
            confusion_matrix += counts.reshape(NUM_CLASSES, NUM_CLASSES)
            
            # 2. 保存前 num_vis 张样本的可视化三合一对比图
            if vis_count < num_vis:
                orig_img_path = dataset.data[step]["image"]
                orig_img = Image.open(orig_img_path).convert("RGB").resize((512, 512), Image.BILINEAR)
                
                # 将预测结果中的忽略区域也涂黑，方便直观对比
                pred_vis_np = pred_np.copy()
                pred_vis_np[target_np == 255] = 255
                
                gt_color = mask_to_color(target_np)
                pred_color = mask_to_color(pred_vis_np)
                
                # 拼接图像：[左：原图 | 中：真实标签 GT | 右：模型预测 Pred]
                concat_img = Image.new("RGB", (512 * 3, 512))
                concat_img.paste(orig_img, (0, 0))
                concat_img.paste(gt_color, (512, 0))
                concat_img.paste(pred_color, (1024, 0))
                
                save_path = f"eval_results/compare_sample_{step}.png"
                concat_img.save(save_path)
                vis_count += 1
                print(f"🎨 已生成可视化对比图 [{vis_count}/{num_vis}]: {save_path}")

    # ==========================================
    # 3. 汇总计算遥感语义分割核心指标
    # ==========================================
    intersection = np.diag(confusion_matrix)
    union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - intersection
    iou_per_class = intersection / (union + 1e-10) * 100
    
    # Overall Accuracy (总体像素精度)
    oa = intersection.sum() / (confusion_matrix.sum() + 1e-10) * 100
    # Mean IoU (平均交并比)
    miou = np.nanmean(iou_per_class)
    
    print("\n" + "="*50)
    print("📊 遥感多模态语义分割定量评估报告")
    print("="*50)
    for i in range(NUM_CLASSES):
        print(f"  - 类别 [{i}] ({CLASS_NAMES[i]:<18}): IoU = {iou_per_class[i]:.2f}%")
    print("-" * 50)
    print(f"🌟 总体精度 (Overall Accuracy, OA) : {oa:.2f}%")
    print(f"🏆 平均交并比 (Mean IoU, mIoU)     : {miou:.2f}%")
    print("="*50)
    print("💡 提示：请打开 eval_results 文件夹查看直观的【原图 | 真实标签 | 模型预测】彩色对比图！")

if __name__ == "__main__":
    # 当训练跑到 2000 步生成检查点后，直接运行 python eval.py 即可检验成果！
    evaluate(checkpoint_dir="checkpoints/step_2000")