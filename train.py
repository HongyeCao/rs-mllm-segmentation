import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from PIL import Image
import numpy as np
import time
import math

# ==========================================
# 1. 定义多模态数据集加载器（含坏图容错机制）
# ==========================================
class RSMultiModalDataset(Dataset):
    def __init__(self, json_path, processor, max_length=1024):
        with open(json_path, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        self.processor = processor
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 加上 try...except 容错机制，防止个别损坏的遥感图像导致 DataLoader 崩溃
        for _ in range(5): # 最多尝试 5 次，如果连续坏图则顺延到下一张
            try:
                item = self.data[idx]
                
                # 读取原图并统一为 512x512
                image = Image.open(item["image"]).convert("RGB").resize((512, 512), Image.BILINEAR)
                
                # 读取掩码
                mask_img = Image.open(item["mask"]).convert("L").resize((512, 512), Image.NEAREST)
                mask_tensor = torch.tensor(np.array(mask_img), dtype=torch.long)
                
                user_text = item["conversations"][0]["value"].replace("<image>\n", "").replace("<image>", "")
                assistant_text = item["conversations"][1]["value"]
                
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "text", "text": user_text}
                        ]
                    },
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": assistant_text}
                        ]
                    }
                ]
                
                prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                
                inputs = self.processor(
                    text=prompt,
                    images=image,
                    return_tensors="pt",
                    padding="max_length",
                    max_length=self.max_length,
                    truncation=True
                )
                
                return {
                    "input_ids": inputs["input_ids"].squeeze(0),
                    "attention_mask": inputs["attention_mask"].squeeze(0),
                    "pixel_values": inputs["pixel_values"].squeeze(0),
                    "image_grid_thw": inputs["image_grid_thw"].squeeze(0),
                    "mm_token_type_ids": inputs["mm_token_type_ids"].squeeze(0),
                    "target_mask": mask_tensor
                }
            except Exception as e:
                # 如果当前索引图片损坏，打印提示并随机换一张加载
                print(f"⚠️ 警告: 索引 {idx} 的图像文件加载损坏 ({e})，正在自动跳过...")
                idx = np.random.randint(0, len(self.data))

# ==========================================
# 2. 定义具备真实空间特征解码的大模型架构
# ==========================================
class RS_MLLM_Segmenter(nn.Module):
    def __init__(self, model_id="Qwen/Qwen2-VL-2B-Instruct", num_classes=6):
        super().__init__()
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True
        )
        
        print("正在加载基础大多模态模型...")
        self.mllm = AutoModelForImageTextToText.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.mllm = prepare_model_for_kbit_training(self.mllm)
        self.mllm.gradient_checkpointing_enable()
        
        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        self.mllm = get_peft_model(self.mllm, peft_config)
        self.mllm.print_trainable_parameters()
        
        # 获取 Qwen2-VL 的隐藏层维度 (1536) 与图像占位符 Token ID
        hidden_size = self.mllm.config.text_config.hidden_size
        self.image_token_id = self.mllm.config.image_token_id
        
        # 视觉空间特征降维投影器
        self.spatial_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # 文本全局语义引导投影器
        self.text_proj = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.GELU(),
            nn.Linear(256, 256)
        )
        
        # 渐进式多级上采样解码器 (将 18x18 空间特征逐步精细还原)
        self.mask_decoder = nn.Sequential(
            # 18x18 -> 36x36
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            # 36x36 -> 72x72
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            # 72x72 -> 144x144
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            # 特征平滑与分类头
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, num_classes, kernel_size=1)
        )

    def forward(self, input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids):
        outputs = self.mllm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            output_hidden_states=True
        )
        
        last_hidden = outputs.hidden_states[-1] # shape: (B, seq_len, hidden_size)
        b = last_hidden.size(0)
        
        spatial_feats = []
        text_feats = []
        
        for i in range(b):
            # 1. 精准提取属于图像的 324 个空间 Token 特征
            img_mask = (input_ids[i] == self.image_token_id)
            img_tokens = last_hidden[i][img_mask] # shape: (324, 1536)
            
            num_tokens = img_tokens.size(0)
            h_grid = int(math.sqrt(num_tokens))
            w_grid = h_grid
            
            # 极端情况兜底保护
            if h_grid * w_grid != num_tokens or num_tokens == 0:
                h_grid, w_grid = 18, 18
                img_tokens = last_hidden[i, :324, :]
                
            img_proj = self.spatial_proj(img_tokens) # (324, 256)
            # 重塑回二维空间网格: (256, 18, 18)
            img_grid = img_proj.transpose(0, 1).contiguous().view(256, h_grid, w_grid)
            spatial_feats.append(img_grid)
            
            # 2. 提取最后一个有效非 Padding Token 作为文本语义引导
            valid_len = int(attention_mask[i].sum().item())
            txt_token = last_hidden[i, valid_len - 1, :]
            txt_proj = self.text_proj(txt_token).view(256, 1, 1) # (256, 1, 1)
            text_feats.append(txt_proj)
            
        spatial_tensor = torch.stack(spatial_feats, dim=0) # (B, 256, 18, 18)
        text_tensor = torch.stack(text_feats, dim=0)       # (B, 256, 1, 1)
        
        # 跨模态融合：利用全局文本语义在空间维度上调制并激活视觉特征图
        fused_feat = spatial_tensor * torch.sigmoid(text_tensor) + spatial_tensor
        
        # 解码器上采样至 144x144
        logits = self.mask_decoder(fused_feat)
        # 双线性插值对齐到标准的 512x512 遥感切片尺寸
        logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)
        
        return logits

def save_checkpoint(model, save_dir):
    """安全分离保存 LoRA 权重与自定义分割解码头"""
    os.makedirs(save_dir, exist_ok=True)
    # 1. 保存轻量化 LoRA 适配器权重
    model.mllm.save_pretrained(os.path.join(save_dir, "lora_weights"))
    # 2. 保存自定义空间投影器与分割解码器权重
    custom_heads = {
        "spatial_proj": model.spatial_proj.state_dict(),
        "text_proj": model.text_proj.state_dict(),
        "mask_decoder": model.mask_decoder.state_dict()
    }
    torch.save(custom_heads, os.path.join(save_dir, "seg_heads.pth"))
    print(f"💾 检查点已安全保存至: {save_dir}")

# ==========================================
# 3. 训练启动逻辑与 NaN 免疫保护
# ==========================================
if __name__ == "__main__":
    BATCH_SIZE = 1
    GRAD_ACCUM_STEPS = 8
    EPOCHS = 10
    LR = 1e-4
    NUM_CLASSES = 6
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")
    
    model = RS_MLLM_Segmenter(num_classes=NUM_CLASSES).to(device)
    
    train_dataset = RSMultiModalDataset("datasets/train_dataset.json", processor)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
    
    model.train()
    print(f"🚀 开始具备真实空间分辨率的多模态训练... 总数据量: {len(train_dataset)}，每轮步数: {len(train_loader)}")
    
    for epoch in range(EPOCHS):
        optimizer.zero_grad()
        running_loss = 0.0
        valid_steps = 0
        start_time = time.time()
        
        for step, batch in enumerate(train_loader):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            image_grid_thw = batch["image_grid_thw"].to(device)
            mm_token_type_ids = batch["mm_token_type_ids"].to(device)
            targets = batch["target_mask"].to(device).long()
            
            # 1. 越界像素过滤：将非法的类别值统一置为 255
            targets[(targets >= NUM_CLASSES) & (targets != 255)] = 255
            
            # 2. 【核心保护】如果整张掩码图全都是 255（无效边缘切片），直接跳过，防止除零产生 NaN！
            if (targets != 255).sum() == 0:
                continue
            
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(input_ids, attention_mask, pixel_values, image_grid_thw, mm_token_type_ids)
                loss = F.cross_entropy(logits, targets, ignore_index=255) / GRAD_ACCUM_STEPS
            
            # 3. 双重保险：万一出现极端数值异常，跳过 backward 避免污染模型权重
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"⚠️ Step {step+1} 检测到异常 Loss (NaN/Inf)，已自动拦截并跳过！")
                continue
                
            loss.backward()
            running_loss += loss.item() * GRAD_ACCUM_STEPS
            valid_steps += 1
            
            # 按有效步数进行梯度累积与参数更新
            if valid_steps > 0 and valid_steps % GRAD_ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()
                
                print(f"Epoch [{epoch+1}/{EPOCHS}] Step [{step+1}/{len(train_loader)}] Loss: {running_loss/GRAD_ACCUM_STEPS:.4f}")
                running_loss = 0.0
                
            # 每 2000 步自动保存一次中途检查点，随时可用于推理测试，无需苦等一整个 Epoch
            if (step + 1) % 2000 == 0:
                save_checkpoint(model, f"checkpoints/step_{step+1}")
                
        # 每个 Epoch 结束保存完整权重
        save_checkpoint(model, f"checkpoints/epoch_{epoch+1}")
        print(f"✅ Epoch {epoch+1} 完成，耗时: {(time.time()-start_time)/60:.2f} 分钟。")