import os
import tarfile
import yaml

# 训练集压缩文件路径
TRAIN_SRC_DIR = '/root/autodl-pub/ImageNet/ILSVRC2012/ILSVRC2012_img_train.tar'
# 解压目标路径
TRAIN_DEST_DIR = '/root/autodl-tmp/imagenet_50classes/train'

def load_config(config_path="config.yaml"):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def extract_train():
    """只解压指定的50个类别"""
    # 加载配置文件
    config = load_config()
    
    # 从配置文件中获取目标类别
    classes_config = config['data']['classes']
    TARGET_CLASSES = set()
    for class_info in classes_config.values():
        TARGET_CLASSES.add(class_info['wnid'])
    
    # 确保目标目录存在
    os.makedirs(TRAIN_DEST_DIR, exist_ok=True)
    
    total_extracted = 0
    extracted_classes = set()
    
    with open(TRAIN_SRC_DIR, 'rb') as f:
        # 第一次打开tar文件，检查类别存在性
        with tarfile.open(fileobj=f, mode='r:') as tar:
            print(f"开始解压 {len(TARGET_CLASSES)} 个目标类别...")
            
            # 首先检查所有目标类别是否在tar文件中
            all_members = tar.getmembers()
            tar_classes = set()
            for member in all_members:
                if member.isfile() and member.name.endswith('.tar'):
                    cls_name = member.name.strip('.tar')
                    tar_classes.add(cls_name)
            
            # 检查缺失的类别
            missing_classes = TARGET_CLASSES - tar_classes
            if missing_classes:
                print(f"\n⚠️  以下 {len(missing_classes)} 个类别在tar文件中不存在:")
                for cls in sorted(missing_classes):
                    print(f"  - {cls}")
            else:
                print("\n✅ 所有目标类别都在tar文件中存在")
    
    # 重新打开tar文件进行解压
    with open(TRAIN_SRC_DIR, 'rb') as f:
        with tarfile.open(fileobj=f, mode='r:') as tar:
            # 开始解压
            for i, item in enumerate(tar):
                if not item.isfile() or not item.name.endswith('.tar'):
                    continue
                    
                cls_name = item.name.strip(".tar")
                
                # 只处理目标类别
                if cls_name not in TARGET_CLASSES:
                    continue
                
                try:
                    # 提取子tar文件
                    a = tar.extractfile(item)
                    if a is None:
                        print(f"❌ 无法提取类别 {cls_name}: 提取文件失败")
                        continue
                    
                    # 打开子tar文件
                    b = tarfile.open(fileobj=a, mode="r:")
                    e_path = f"{TRAIN_DEST_DIR}/{cls_name}/"
                    
                    # 创建目标目录
                    if not os.path.isdir(e_path):
                        os.makedirs(e_path, exist_ok=True)
                    
                    # 解压文件
                    print(f"# {i} 解压类别 {cls_name} 到 >>> {e_path}")
                    b.extractall(e_path)
                    
                    # 记录已解压的类别
                    extracted_classes.add(cls_name)
                    total_extracted += 1
                    
                    # 关闭文件
                    b.close()
                    a.close()
                    
                except Exception as e:
                    print(f"❌ 解压类别 {cls_name} 时出错: {e}")
                    continue
            
            # 检查未解压的类别
            not_extracted = TARGET_CLASSES - extracted_classes
            if not_extracted:
                print(f"\n⚠️  以下 {len(not_extracted)} 个类别未成功解压:")
                for cls in sorted(not_extracted):
                    print(f"  - {cls}")
            
            print(f"\n✅ 完成！共解压 {total_extracted} 个类别")

if __name__ == '__main__':
    extract_train()
