import os
import yaml
import shutil
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
from diffusers import AutoencoderKL
from tqdm import tqdm

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

class ImageNetSubset(Dataset):
    def __init__(self, root_dir, classes_config, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.labels = []
        
        for class_name, class_info in classes_config.items():
            folder_name = class_info['wnid']
            class_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(class_path):
                for img_name in os.listdir(class_path):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append(os.path.join(class_path, img_name))
                        self.labels.append(class_name)
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        class_name = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, class_name

def copy_images_to_output(config):
    """从原始 ImageNet 复制图片到输出目录，用于后续 FID"""
    source_root = config['data']['image_root']
    dest_root = config['evaluating']['real_images']
    classes_config = config['data']['classes']
    
    # 检查源目录和目标目录是否相同
    if os.path.abspath(source_root) == os.path.abspath(dest_root):
        print(f">>> 源目录和目标目录相同 ({source_root})，跳过复制操作")
        return
    
    os.makedirs(dest_root, exist_ok=True)
    print(f">>> 正在从 {source_root} 复制图片到 {dest_root}")
    
    total_copied = 0
    for class_name, class_info in classes_config.items():
        folder_name = class_info['wnid']
        source_dir = os.path.join(source_root, folder_name)
        dest_dir = os.path.join(dest_root, folder_name)
        
        if not os.path.exists(source_dir):
            print(f"  警告: {source_dir} 不存在，跳过")
            continue
            
        os.makedirs(dest_dir, exist_ok=True)
        
        img_count = 0
        for img_name in os.listdir(source_dir):
            if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                src_path = os.path.join(source_dir, img_name)
                dst_path = os.path.join(dest_dir, img_name)
                # 检查源文件和目标文件是否相同
                if os.path.abspath(src_path) != os.path.abspath(dst_path):
                    shutil.copy2(src_path, dst_path)
                    img_count += 1
                
        total_copied += img_count
        print(f"  {class_name}: 复制 {img_count} 张")
    
    print(f"✅ 共复制 {total_copied} 张图片\n")

def load_vae(config):
    """加载 VAE 模型"""
    vae_config = config['models']['vae']
    vae = AutoencoderKL.from_pretrained(vae_config['pretrained_path'])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    return vae, vae_config['scale_factor']

def get_data_transforms(is_train=True):
    """获取数据变换，包含数据增强"""
    if is_train:
        return transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(256),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

def prepare_latents(config):
    """准备 latents 数据"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f">>> 使用设备: {device}")
    
    # 数据变换
    transform = get_data_transforms(is_train=True)
    
    # 数据集加载
    classes_config = config['data']['classes']
    dataset = ImageNetSubset(config['data']['image_root'], classes_config, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    print(f"数据集大小: {len(dataset)}")
    if len(dataset) == 0:
        raise ValueError("数据集为空，请检查路径和文件夹名称是否正确！")
    
    # 加载 VAE
    vae, scale_factor = load_vae(config)
    vae = vae.to(device)
    
    # 提取 latents
    print(">>> 开始提取 Latents...")
    all_latents = []
    all_labels = []
    
    for batch_imgs, batch_class_names in tqdm(dataloader, desc="Processing batches"):
        batch_imgs = batch_imgs.to(device)
        
        # 转换 class_names 为 class_ids
        batch_label_ids = []
        for class_name in batch_class_names:
            batch_label_ids.append(config['data']['classes'][class_name]['id'])
        batch_labels = torch.tensor(batch_label_ids)
        
        with torch.no_grad():
            posterior = vae.encode(batch_imgs).latent_dist
            latents = posterior.sample() * scale_factor
        
        all_latents.append(latents.cpu())
        all_labels.append(batch_labels)
    
    final_latents = torch.cat(all_latents, dim=0)
    final_labels = torch.cat(all_labels, dim=0)
    
    # 保存 latents
    save_path = config['models']['vae']['latent_path']
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save({
        'latents': final_latents,
        'labels': final_labels
    }, save_path)
    
    print(f"✅ Latents 已保存至: {save_path}")
    print(f"  Latents shape: {final_latents.shape}")
    print(f"  Labels shape: {final_labels.shape}")

def main():
    config = load_config()
    
    # 直接生成 latents
    print("=" * 50)
    print("生成 VAE Latents")
    print("=" * 50)
    prepare_latents(config)
    
    print("\n✅ 所有步骤完成！")

if __name__ == "__main__":
    main()