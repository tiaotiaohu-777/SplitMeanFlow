import os
import sys
import yaml
from cleanfid import fid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    SAVE_DIR = "/root/autodl-tmp/teacher_samples_50NFEs_4cfg"
    NFE = 50
    CFG_SCALE = 4.0

    print(">>> 开始计算 FID...")
    real_images_path = config['evaluating']['real_images']
    fid_value = fid.compute_fid(
        real_images_path,
        SAVE_DIR,
        batch_size=64,
        num_workers=8
    )
    fid_msg = f"[Teacher FID] NFE: {NFE:02d} | CFG: {CFG_SCALE:2.1f} | FID: {fid_value:8.2f}\n"
    print(f"\n🎯 [Teacher] FID = {fid_value:.2f}")

    with open(config['log'], "a", encoding="utf-8") as f:
        f.write(fid_msg)

    print("\n>>> FID 计算完成！")

if __name__ == "__main__":
    main()