import os
import sys
import yaml
import math
import datetime
import copy
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models import SiT_XL_2

def update_ema(ema_model, model, decay=0.9999):
    with torch.no_grad():
        for ema_param, param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

def perform_surgery(model, ckpt_path, original_ids):
    print(f">>> 加载预训练权重: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state_dict = checkpoint.get('ema', checkpoint)

    embed_key = 'y_embedder.embedding_table.weight' if 'y_embedder.embedding_table.weight' in state_dict else 'y_embedder.weight'
    old_weight = state_dict[embed_key]

    class_weights = old_weight[torch.tensor(original_ids, dtype=torch.long)]
    null_weight = old_weight[-1].unsqueeze(0)

    state_dict[embed_key] = torch.cat([class_weights, null_weight], dim=0)
    model.load_state_dict(state_dict, strict=True)
    print("✅ 权重外科手术完成！")

def train():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    CLASS_DICT = config['data']['classes']
    ORIGINAL_IDS = [class_info['id'] for class_info in CLASS_DICT.values()]
    NUM_CLASSES = len(ORIGINAL_IDS)

    LATENTS_PATH = config['models']['vae']['latent_path']
    LOG_FILE = config['log']
    STAGE1_CFG = config['stage1_teacher']

    BATCH_SIZE = STAGE1_CFG['batch_size']
    LR = float(STAGE1_CFG['lr'])
    TOTAL_STEPS = STAGE1_CFG['total_steps']
    PRETRAINED_WEIGHTS_PATH = STAGE1_CFG['pretrained_path']
    SAVE_DIR = STAGE1_CFG['save_path']
    FREEZE_LAYERS = 20

    os.makedirs(SAVE_DIR, exist_ok=True)
    print(f"🚀 Stage 1 教师训练: BS={BATCH_SIZE}, LR={LR}, 步数={TOTAL_STEPS}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dict = torch.load(LATENTS_PATH, map_location="cpu")
    latents = data_dict['latents'].float()
    original_labels = data_dict['labels'].long()

    print(f"    Latents 平均方差: {latents.var().item():.4f}")

    id_to_idx = {orig_id: idx for idx, orig_id in enumerate(ORIGINAL_IDS)}
    mapped_labels = torch.zeros_like(original_labels)
    for orig_id, new_idx in id_to_idx.items():
        mapped_labels[original_labels == orig_id] = new_idx

    dataset = TensorDataset(latents, mapped_labels)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    model = SiT_XL_2(num_classes=NUM_CLASSES).to(device)
    perform_surgery(model, PRETRAINED_WEIGHTS_PATH, ORIGINAL_IDS)
    model.train()

    for i, block in enumerate(model.blocks):
        if i < FREEZE_LAYERS:
            for param in block.parameters():
                param.requires_grad = False

    ema_model = copy.deepcopy(model).eval()
    for param in ema_model.parameters():
        param.requires_grad = False

    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LR, weight_decay=1e-4, fused=True)
    scaler = GradScaler()

    warmup_steps = int(TOTAL_STEPS * 0.05)
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, TOTAL_STEPS - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = LambdaLR(optimizer, lr_lambda)

    global_step = 0
    progress_bar = tqdm(total=TOTAL_STEPS, desc="Teacher Finetuning")

    loss_history = []
    step_history = []

    with open(LOG_FILE, "w", encoding="utf-8") as f:
        f.write("=== SplitMeanFlow Stage 1: Teacher Finetuning ===\n")

    while global_step < TOTAL_STEPS:
        for (x1, y) in dataloader:
            if global_step >= TOTAL_STEPS:
                break

            optimizer.zero_grad()
            x1, y = x1.to(device), y.to(device)
            B = x1.shape[0]

            drop_ids = torch.rand(B, device=device) < 0.1
            y = torch.where(drop_ids, NUM_CLASSES, y)

            x0 = torch.randn_like(x1)
            t = torch.rand((B,), device=device).view(B, 1, 1, 1)
            xt = t * x1 + (1 - t) * x0
            target_v = x1 - x0

            with autocast():
                loss = F.mse_loss(model(xt, t.view(B), y), target_v)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()

            scheduler.step()
            update_ema(ema_model, model)
            global_step += 1
            progress_bar.update(1)

            if global_step % 50 == 0:
                time_str = datetime.datetime.now().strftime("%m/%d/%H/%M/%S")
                cur_lr = optimizer.param_groups[0]['lr']
                loss_value = loss.item()
                log_str = f"[teacher] {time_str} step:{global_step:06d} loss:{loss_value:.6f} lr:{cur_lr:.6e}\n"

                loss_history.append(loss_value)
                step_history.append(global_step)

                with open(LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(log_str)
                progress_bar.set_postfix({"Loss": f"{loss_value:.4f}", "LR": f"{cur_lr:.2e}"})

            if global_step % 1000 == 0:
                torch.save(ema_model.state_dict(), os.path.join(SAVE_DIR, f"teacher_ema_step_{global_step}.pt"))

    print("\n🎉 教师训练结束！")

    # 绘制并保存损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(step_history, loss_history, label="Training Loss")
    plt.title("Teacher Model Training Loss")
    plt.xlabel("Steps")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.legend()
    
    # 保存到主文件夹
    loss_curve_path = os.path.join(os.path.dirname(SAVE_DIR), "teacher_loss_curve.png")
    plt.savefig(loss_curve_path)
    print(f"Loss curve saved to: {loss_curve_path}")

if __name__ == "__main__":
    train()