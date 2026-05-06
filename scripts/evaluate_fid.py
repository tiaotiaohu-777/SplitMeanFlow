import os
import re
import yaml
from cleanfid import fid

def evaluate_fid():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    real_images_path = config['evaluating']['real_images']
    generated_images_base = config['evaluating']['generated_images']
    log_path = config['log']

    subdirs = sorted([d for d in os.listdir(generated_images_base)
                      if os.path.isdir(os.path.join(generated_images_base, d))])

    results = []

    for subdir in subdirs:
        generated_images_path = os.path.join(generated_images_base, subdir)

        match = re.match(r'(\d+)NFEs_(\d+)cfg', subdir)
        if match:
            nfe = match.group(1)
            cfg = match.group(2)
            desc = f"NFE={nfe:>2} CFG={cfg:>2}"
        else:
            desc = subdir

        print(f"正在评估: {desc} ...")

        fid_value = fid.compute_fid(
            real_images_path,
            generated_images_path,
            batch_size=64,
            num_workers=8
        )

        results.append((desc, fid_value))
        log_msg = f"FID: {fid_value:8.2f} | {desc} | Path: {generated_images_path}\n"

        with open(log_path, "a", encoding="utf-8") as f:
            f.write(log_msg)

        print(f"FID: {fid_value:.2f} | {desc}")

    print("\n" + "="*60)
    print("FID 评估结果汇总:")
    print("="*60)
    for desc, fid_value in results:
        print(f"  {desc}: {fid_value:.2f}")
    print("="*60)

    best_result = min(results, key=lambda x: x[1])
    print(f"\n最佳配置: {best_result[0]} (FID={best_result[1]:.2f})")

if __name__ == "__main__":
    evaluate_fid()