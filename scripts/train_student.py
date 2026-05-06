import os
import sys
import yaml
import math
import copy
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models import SiT_XL_2, SiT_XL_2_MeanFlow
from src.splitmeanflow import SplitMeanFlow

torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_math_sdp(False)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def init_model(config, device):
    num_classes = len(config['data']['classes'])
    freeze_layers = config['stage2_student'].get('freeze_layers', 0)
    teacher_ckpt_path = config['stage2_student']['teacher_path']

    # 初始化教师模型
    teacher_model = SiT_XL_2(num_classes=num_classes).to(device)
    with torch.no_grad():
        state_dict = torch.load(teacher_ckpt_path, map_location=device)
        teacher_model.load_state_dict(state_dict, strict=True)
        teacher_model.eval()

    # 初始化学生模型
    student_model = SiT_XL_2_MeanFlow(num_classes=num_classes).to(device)
    student_model.load_state_dict(state_dict, strict=False)
    with torch.no_grad():
        nn.init.constant_(student_model.r_embedder.mlp[-1].weight, 0)
        nn.init.constant_(student_model.r_embedder.mlp[-1].bias, 0)
    
    # 冻结指定层数
    if freeze_layers > 0:
        for i, block in enumerate(student_model.blocks):
            if i < freeze_layers:
                for param in block.parameters():
                    param.requires_grad = False
    
    # 确保关键部分可训练
    for param in student_model.r_embedder.parameters():
        param.requires_grad = True
    for param in student_model.final_layer.parameters():
        param.requires_grad = True
    
    student_model.train()
    return teacher_model, student_model

def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    latent_path = config["models"]["vae"]["latent_path"]
    data_dict = torch.load(latent_path, map_location="cpu")
    latents = data_dict['latents'].float()
    original_labels = data_dict['labels'].long()

    class_dict = config['data']['classes']
    original_ids = [class_info['id'] for class_info in class_dict.values()]
    num_classes = len(original_ids)

    id_to_idx = {orig_id: idx for idx, orig_id in enumerate(original_ids)}
    mapped_labels = torch.zeros_like(original_labels)
    for orig_id, new_idx in id_to_idx.items():
        mapped_labels[original_labels == orig_id] = new_idx

    dataset = TensorDataset(latents, mapped_labels)
    batch_size = config["stage2_student"]["batch_size"]
    accumulate_steps = config["stage2_student"]["accumulate_steps"]

    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=8, pin_memory=True, prefetch_factor=2, drop_last=True
    )

    print("Initializing Models...")
    teacher_model, student_model = init_model(config, device)

    ema_model = copy.deepcopy(student_model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    # 优化器和学习率调度器
    lr = float(config["stage2_student"]["lr"])
    total_steps = config["stage2_student"]["total_steps"]
    warmup_steps = int(total_steps * 0.05)
    optimizer = AdamW(student_model.parameters(), lr=lr, weight_decay=1e-4, fused=True)
    scaler = GradScaler()
    
    # 损失权重配置
    loss_weights = config["stage2_student"].get("loss_weights", {"boundary": 1.0, "split": 1.0})
    boundary_weight = loss_weights.get("boundary", 1.0)
    split_weight = loss_weights.get("split", 1.0)

    def lr_lambda(step):
        # step是权重更新次数，需要考虑梯度累计
        if step < warmup_steps / accumulate_steps:
            return float(step) / max(1, warmup_steps / accumulate_steps)
        elif step < int(total_steps * 0.8 / accumulate_steps):
            return 1.0
        else:
            progress = float(step - int(total_steps * 0.8 / accumulate_steps)) / (total_steps / accumulate_steps - int(total_steps * 0.8 / accumulate_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    split_flow = SplitMeanFlow()

    save_dir = config["stage2_student"]["save_path"]
    os.makedirs(save_dir, exist_ok=True)

    # 训练历史
    loss_history = {"split": [], "bound": [], "total": []}
    step_history = []

    best_loss = float('inf')
    best_step = 0

    print(f"\n🚀 Starting Stage 2 Distillation (BS={batch_size}, LR={lr}, Steps={total_steps})")

    global_step = 0
    progress_bar = tqdm(total=total_steps, desc="Student Distillation")

    while global_step < total_steps:
        for x_batch, c_batch in dataloader:
            if global_step >= total_steps:
                break

            x_batch = x_batch.to(device, non_blocking=True)
            c_batch = c_batch.to(device, non_blocking=True)

            # 前向传播
            with autocast():
                loss_bound = split_flow.BoundaryLoss(student_model, teacher_model, x_batch, c_batch)
                loss_split = split_flow.SplitLoss(student_model, ema_model, x_batch, c_batch)
                loss = split_weight * loss_split + boundary_weight * loss_bound

            # 反向传播（简化的梯度累计）
            scaler.scale(loss).backward()

            # 累计步数后更新参数
            if (global_step + 1) % accumulate_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                # 更新EMA模型
                if global_step > warmup_steps:
                    update_ema(ema_model, student_model)

            # 记录损失
            if global_step % 100 == 0:
                cur_lr = optimizer.param_groups[0]['lr']
                ls_val = loss_split.item()
                lb_val = loss_bound.item()
                lt_val = loss.item()

                loss_history["split"].append(ls_val)
                loss_history["bound"].append(lb_val)
                loss_history["total"].append(lt_val)
                step_history.append(global_step)

                log_str = f"step:{global_step:06d} | L_total:{lt_val:.6f} (Split:{ls_val:.6f}, Bound:{lb_val:.6f}) | lr:{cur_lr:.2e}\n"
                with open(config.get("log", "training.log"), "a", encoding="utf-8") as f:
                    f.write(log_str)
                print(log_str.strip())

                if lt_val < best_loss:
                    best_loss = lt_val
                    best_step = global_step
                    torch.save(ema_model.state_dict(), os.path.join(save_dir, "student_ema_best.pt"))
                    print(f"  💾 New best model saved! Loss: {best_loss:.6f} at step {best_step}")

            global_step += 1
            progress_bar.update(1)

    torch.save(ema_model.state_dict(), os.path.join(save_dir, "student_ema_final.pt"))
    print("\n✅ Stage 2 Training Complete!")
    print(f"📊 Best model: step {best_step} with loss {best_loss:.6f}")
    print(f"💾 Best model saved to: {os.path.join(save_dir, 'student_ema_best.pt')}")
    print(f"💾 Final model saved to: {os.path.join(save_dir, 'student_ema_final.pt')}")

    # 绘制损失曲线
    plt.figure(figsize=(12, 6))
    plt.plot(step_history, loss_history["total"], label="Total Loss", color='black', linewidth=2)
    plt.plot(step_history, loss_history["split"], label="Split Loss", color='red', alpha=0.8)
    plt.plot(step_history, loss_history["bound"], label="Boundary Loss", color='blue', alpha=0.5)
    plt.title("SplitMeanFlow Stage 2 Distillation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss Value")
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(save_dir, "student_loss_curve.png"))
    print(f"Saved loss curve to {os.path.join(save_dir, 'student_loss_curve.png')}")

if __name__ == "__main__":
    train()