"""
可视化ResNet50的特征层级
visualize_resnet_layers.py
"""
import torch
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torchvision.transforms as T


def load_and_preprocess_image(image_path='sample.jpg'):
    """加载并预处理图像"""
    # 如果没有图像，创建一个随机图像
    if image_path == 'sample.jpg':
        img = Image.new('RGB', (320, 320), color=(128, 128, 128))
        # 画一些简单的形状
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)
        draw.rectangle([50, 50, 150, 150], fill=(255, 0, 0))
        draw.ellipse([200, 200, 280, 280], fill=(0, 255, 0))
    else:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((320, 320))
    
    # 转换为tensor
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225])
    ])
    
    return transform(img).unsqueeze(0), img


def visualize_multiscale_features():
    """可视化多尺度特征"""
    print("="*70)
    print("  可视化ResNet50多尺度特征")
    print("="*70)
    
    # 加载模型
    resnet = models.resnet50(pretrained=True)
    resnet.eval()
    
    # 提取层级
    layer0 = torch.nn.Sequential(
        resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool
    )
    layer1 = resnet.layer1
    layer2 = resnet.layer2
    layer3 = resnet.layer3
    
    # 加载图像
    x, orig_img = load_and_preprocess_image()
    
    print(f"\n输入图像: {x.shape}")
    
    # 提取特征
    with torch.no_grad():
        x0 = layer0(x)
        f1 = layer1(x0)
        f2 = layer2(f1)
        f3 = layer3(f2)
    
    print(f"Layer1特征: {f1.shape}")
    print(f"Layer2特征: {f2.shape}")
    print(f"Layer3特征: {f3.shape}")
    
    # ═══════════════════════════════════════════
    # 可视化
    # ═══════════════════════════════════════════
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    
    # 第一行：原图和特征图
    axes[0, 0].imshow(orig_img)
    axes[0, 0].set_title('原始图像\n(320×320)', fontsize=10)
    axes[0, 0].axis('off')
    
    features = [f1, f2, f3]
    titles = [
        'Layer1特征\n(256通道, 80×80)',
        'Layer2特征\n(512通道, 40×40)',
        'Layer3特征\n(1024通道, 20×20)'
    ]
    
    for i, (feat, title) in enumerate(zip(features, titles)):
        # 可视化前16个通道的平均
        vis = feat[0, :16].mean(0).cpu().numpy()
        axes[0, i+1].imshow(vis, cmap='viridis')
        axes[0, i+1].set_title(title, fontsize=10)
        axes[0, i+1].axis('off')
    
    # 第二行：每层的通道激活统计
    axes[1, 0].text(0.5, 0.5, '通道激活\n统计', 
                    ha='center', va='center', fontsize=12)
    axes[1, 0].axis('off')
    
    for i, (feat, title) in enumerate(zip(features, titles)):
        # 计算每个通道的平均激活
        channel_activations = feat[0].mean(dim=[1, 2]).cpu().numpy()
        
        axes[1, i+1].hist(channel_activations, bins=50, alpha=0.7)
        axes[1, i+1].set_title(f'{title.split()[0]}\n通道激活分布', fontsize=10)
        axes[1, i+1].set_xlabel('激活值')
        axes[1, i+1].set_ylabel('通道数')
    
    plt.tight_layout()
    plt.savefig('resnet50_multiscale_features.png', dpi=150, bbox_inches='tight')
    print("\n可视化已保存到: resnet50_multiscale_features.png")
    plt.show()


def visualize_information_flow():
    """可视化信息流"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 定义层级
    layers = [
        ('输入图像', 3, 320, 320),
        ('Layer0', 64, 80, 80),
        ('Layer1', 256, 80, 80),
        ('Layer2', 512, 40, 40),
        ('Layer3', 1024, 20, 20),
        ('Layer4\n(不用)', 2048, 10, 10)
    ]
    
    y_positions = np.linspace(0.9, 0.1, len(layers))
    
    for i, (name, channels, h, w) in enumerate(layers):
        y = y_positions[i]
        
        # 画框
        if i < 5:
            color = 'lightblue' if i <= 4 else 'lightgray'
            alpha = 1.0 if i <= 4 else 0.3
        else:
            color = 'lightgray'
            alpha = 0.3
        
        rect = plt.Rectangle((0.2, y-0.05), 0.6, 0.08, 
                             facecolor=color, alpha=alpha, 
                             edgecolor='black', linewidth=2)
        ax.add_patch(rect)
        
        # 标签
        ax.text(0.5, y, name, ha='center', va='center', 
               fontsize=12, fontweight='bold')
        ax.text(0.85, y, f'{channels}通道\n{h}×{h}', 
               ha='left', va='center', fontsize=10)
        
        # 画箭头
        if i < len(layers) - 1:
            ax.annotate('', xy=(0.5, y_positions[i+1]+0.04), 
                       xytext=(0.5, y-0.05),
                       arrowprops=dict(arrowstyle='->', lw=2, 
                                     color='gray' if i >= 4 else 'black'))
    
    # 标记多尺度使用的层
    for i in [2, 3, 4]:  # layer1, 2, 3
        y = y_positions[i]
        ax.text(0.05, y, '⭐', fontsize=20, ha='center', va='center')
    
    ax.text(0.05, 0.95, '多尺度\n使用', ha='center', va='center', 
           fontsize=10, fontweight='bold')
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('ResNet50信息流（多尺度使用layer1,2,3）', 
                fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    plt.savefig('resnet50_info_flow.png', dpi=150, bbox_inches='tight')
    print("信息流图已保存到: resnet50_info_flow.png")
    plt.show()


if __name__ == '__main__':
    print("\n1. 可视化多尺度特征...")
    visualize_multiscale_features()
    
    print("\n2. 可视化信息流...")
    visualize_information_flow()
    
    print("\n" + "="*70)
    print("  完成！")
    print("="*70)