# ==============================================================================
# 脚本名称：inference_single.py
# 作用：加载训练好的检查点，对任意一张新的遥感影像进行端到端推理，并输出可视化彩色预测图
# ==============================================================================

import os
import torch
import torch.nn.functional as F
from transformers import AutoProcessor
from PIL import Image
import numpy as np

# 复用 train.py 中的模型定义与 eval.py 中的色彩转换函数
from train import RS_MLLM_Segmenter
from eval import load_trained_model, mask_to_color

def predict_single_image(image_path, checkpoint_dir="checkpoints/step_2000", output_path="eval_results/prediction_demo.png"):
    """
    对单张遥感影像进行多模态大模型语义分割推理
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    # 加载训练好的完整模型
    model = load_trained_model(checkpoint_dir, num_classes=6, device=device)
    
    # 读取并预处理输入影像
    print(f"📷 正在读取输入遥感影像: {image_path}")
    orig_image = Image.open(image_path).convert("RGB")
    w, h = orig_image.size
    
    # 调整为训练标准的 512x512 输入
    image_resized = orig_image.resize((512, 512), Image.BILINEAR)
    
    prompt = "<image>\n请精确识别并提取该高分遥感影像中的像素级地物分布掩码。"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
    text_prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    inputs = processor(
        text=[text_prompt],
        images=[image_resized],
        return_tensors="pt",
        padding=True
    )
    
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device, dtype=torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"].to(device)
    mm_token_type_ids = inputs["mm_token_type_ids"].to(device)
    
    print("🚀 正在通过多模态大模型及空间解码器进行前向推理...")
    with torch.no_grad():
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = model(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
            # 双线性插值还原回原图尺寸 (H, W)
            logits = F.interpolate(logits, size=(h, w), mode="bilinear", align_corners=False)
            preds = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
    # 将预测索引图转换为科研标准彩色 RGB 图像
    pred_color_img = mask_to_color(preds)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pred_color_img.save(output_path)
    print(f"🎉 推理完成！彩色语义分割预测图已成功保存至: {output_path}")

if __name__ == "__main__":
    # 测试推理单张图片（确保该路径下存在一张测试图）
    test_img = "datasets/train/img/sample.png"
    if os.path.exists(test_img):
        predict_single_image(test_img, checkpoint_dir="checkpoints/step_2000")
    else:
        print("💡 提示：请将 test_img 变量修改为您本地真实的某张测试遥感影像路径。")
