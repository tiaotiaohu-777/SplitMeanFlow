import os
import sys
import time
import yaml
import torch
from torchvision.utils import save_image
from diffusers.models import AutoencoderKL
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.models import SiT_XL_2_MeanFlow

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    samp_cfg = config["sampling"]
    CLASS_IDS = [class_info['id'] for class_info in config['data']['classes'].values()]
    NUM_CLASSES = len(CLASS_IDS)
    BS = samp_cfg["batch_size"]
    NUM_PER_CLASS = samp_cfg["num_samples_per_class"]
    SAVE_DIR_BASE = samp_cfg["save_dir"]
    NFEs = samp_cfg["NFEs"]
    cfg_scales = samp_cfg["cfg_scales"]

    print(">>> 加载 Student 模型与 VAE...")

    vae = AutoencoderKL.from_pretrained(config['models']['vae']['pretrained_path']).to(device)
    vae.eval()
    vae.requires_grad_(False)
    scale_factor = config['models']['vae']['scale_factor']

    student_model = SiT_XL_2_MeanFlow(num_classes=NUM_CLASSES).to(device)
    student_model.load_state_dict(torch.load(samp_cfg["ckpt_path"], map_location=device))
    student_model.eval()
    student_model.requires_grad_(False)

    torch.backends.cuda.enable_flash_sdp(True)

    print(">>> 预热 GPU...")
    with torch.no_grad(), torch.cuda.amp.autocast():
        _z = torch.randn(2, 4, 32, 32, device=device)
        _t = torch.ones(2, device=device)
        _y = torch.tensor([0, 0], device=device)
        _ = student_model(_z, _t, _y, _t)

    for nfe in NFEs:
        for cfg_scale in cfg_scales:
            save_dir = os.path.join(SAVE_DIR_BASE, f"{nfe}NFEs_{cfg_scale}cfg")
            os.makedirs(save_dir, exist_ok=True)
            total_images_generated = 0
            dt = 1.0 / nfe

            start_time = time.time()

            with torch.no_grad(), torch.cuda.amp.autocast():
                for cls_idx, real_cls_id in enumerate(CLASS_IDS):
                    generated_for_this_class = 0
                    pbar = tqdm(total=NUM_PER_CLASS, desc=f"NFE={nfe} | CFG={cfg_scale} | Class {real_cls_id}")

                    while generated_for_this_class < NUM_PER_CLASS:
                        current_bs = min(BS, NUM_PER_CLASS - generated_for_this_class)

                        z_t = torch.randn(current_bs, 4, 32, 32, device=device)
                        y = torch.full((current_bs,), cls_idx, dtype=torch.long, device=device)

                        for step in range(nfe):
                            t_val = step * dt
                            r_val = t_val + dt
                            t_ = torch.full((current_bs,), t_val, device=device)
                            r_ = torch.full((current_bs,), r_val, device=device)

                            if cfg_scale == 1:
                                u = student_model(z_t, t_, y, r_)
                            else:
                                combined_z = torch.cat([z_t, z_t], dim=0)
                                combined_t = torch.cat([t_, t_], dim=0)
                                combined_r = torch.cat([r_, r_], dim=0)
                                uncond_y = torch.full((current_bs,), NUM_CLASSES, dtype=torch.long, device=device)
                                combined_y = torch.cat([y, uncond_y], dim=0)

                                model_out = student_model(combined_z, combined_t, combined_y, combined_r)
                                cond_out, uncond_out = model_out.chunk(2, dim=0)
                                u = uncond_out + cfg_scale * (cond_out - uncond_out)

                            z_t = z_t + u * dt

                        z_0 = z_t / scale_factor
                        images = vae.decode(z_0).sample
                        images = (images / 2 + 0.5).clamp(0, 1)

                        for i in range(current_bs):
                            img_idx = generated_for_this_class + i
                            save_path = os.path.join(save_dir, f"class_{real_cls_id}_img_{img_idx}.png")
                            save_image(images[i], save_path)

                        generated_for_this_class += current_bs
                        total_images_generated += current_bs
                        pbar.update(current_bs)

                    pbar.close()

            total_time = time.time() - start_time
            fps = total_images_generated / total_time

            log_msg = f"NFE: {nfe:02d} | CFG: {cfg_scale:2d} | Total: {total_images_generated:5d} imgs | Time: {total_time:7.2f} s | FPS: {fps:7.2f} imgs/s\n"
            print(f"\n📊 NFE={nfe:02d} CFG={cfg_scale:2d} 结果: {total_time:.2f}s, {fps:.2f} imgs/s")

            with open(config['log'], "a", encoding="utf-8") as f:
                f.write(log_msg)

if __name__ == "__main__":
    main()