"""
GSVCitiesDataloader.py - 修改版（支持旋转数据增强）
用于消融实验：对比 MixVPR+旋转增强 vs DR-VPR
"""
import torch
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from . import PittsburgDataset
from dataloaders.ConPRDataset import ConPRDataset
from dataloaders.ConPRValidation import get_conpr_validation_set
from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL',
    'Rome', 'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix',
    'TRT', 'Boston', 'Lisbon', 'Medellin', 'Minneapolis', 'PRG',
    'WashingtonDC', 'Brussels', 'London', 'Melbourne', 'Osaka', 'PRS',
]


# ============================================================
# 新增：旋转增强类
# ============================================================
class RandomRotationWithProb:
    """
    以一定概率应用随机旋转
    
    参数:
        degrees: 旋转角度范围，(0, 360) 表示 0° 到 360° 之间随机旋转
        p: 应用旋转的概率，默认 0.5 (50%)
        fill: 旋转后空白区域的填充值，默认 0 (黑色)
    """
    def __init__(self, degrees=(0, 360), p=0.5, fill=0):
        self.rotation = T.RandomRotation(
            degrees=degrees,
            interpolation=T.InterpolationMode.BILINEAR,
            fill=fill
        )
        self.p = p
        self.degrees = degrees
    
    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            return self.rotation(img)
        return img
    
    def __repr__(self):
        return f"RandomRotationWithProb(degrees={self.degrees}, p={self.p})"


class GSVCitiesDataModule(pl.LightningDataModule):
    def __init__(self,
                 batch_size=32,
                 img_per_place=4,
                 min_img_per_place=4,
                 shuffle_all=False,
                 image_size=(480, 640),
                 num_workers=4,
                 show_data_stats=True,
                 cities=TRAIN_CITIES,
                 mean_std=IMAGENET_MEAN_STD,
                 batch_sampler=None,
                 random_sample_from_each_place=True,
                 val_set_names=['pitts30k_val'],
                 conpr_sequences=None,
                 conpr_yaw_threshold=80.0,
                 # ========== 新增参数 ==========
                 use_rotation_augmentation=False,  # 是否启用旋转增强
                 rotation_probability=0.5,         # 旋转概率 (50%)
                 rotation_degrees=(0, 360),        # 旋转角度范围
                 ):
        super().__init__()
        self.conpr_sequences = conpr_sequences
        self.conpr_yaw_threshold = conpr_yaw_threshold
        self.batch_size = batch_size
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.shuffle_all = shuffle_all
        self.image_size = image_size
        self.num_workers = num_workers
        self.batch_sampler = batch_sampler
        self.show_data_stats = show_data_stats
        self.cities = cities
        self.mean_dataset = mean_std['mean']
        self.std_dataset = mean_std['std']
        self.random_sample_from_each_place = random_sample_from_each_place
        self.val_set_names = val_set_names
        
        # 新增：旋转增强配置
        self.use_rotation_augmentation = use_rotation_augmentation
        self.rotation_probability = rotation_probability
        self.rotation_degrees = rotation_degrees
        
        self.save_hyperparameters()

        # ========== 构建训练 transform ==========
        train_transform_list = [
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
        ]
        
        # 如果启用旋转增强，在 Resize 后添加
        if use_rotation_augmentation:
            train_transform_list.append(
                RandomRotationWithProb(
                    degrees=rotation_degrees,
                    p=rotation_probability,
                    fill=0
                )
            )
            print(f"✓ 旋转数据增强已启用: {rotation_probability*100:.0f}% 概率, {rotation_degrees[0]}°-{rotation_degrees[1]}° 范围")
        else:
            print("✗ 旋转数据增强未启用")
        
        # 添加其他增强
        train_transform_list.extend([
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])
        
        self.train_transform = T.Compose(train_transform_list)

        # 验证集 transform（不使用旋转增强）
        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all
        }

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers // 2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False
        }

    def setup(self, stage):
        if stage == 'fit':
            self.reload()

            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == 'pitts30k_test':
                    self.val_datasets.append(PittsburgDataset.get_whole_test_set(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'pitts30k_val':
                    self.val_datasets.append(PittsburgDataset.get_whole_val_set(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'conpr':
                    conpr_dataset = get_conpr_validation_set(
                        sequences=self.conpr_sequences,
                        transform=self.valid_transform,
                        yaw_threshold=self.conpr_yaw_threshold
                    )
                    self.val_datasets.append(conpr_dataset)
                else:
                    print(f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
                    
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform
        )

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders

    def print_stats(self):
        print()
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["# of cities", f"{len(TRAIN_CITIES)}"])
        table.add_row(["# of places", f'{self.train_dataset.__len__()}'])
        table.add_row(["# of images", f'{self.train_dataset.total_nb_images}'])
        # 新增：显示旋转增强状态
        if self.use_rotation_augmentation:
            table.add_row(["Rotation Aug", f"✓ {self.rotation_probability*100:.0f}% prob, {self.rotation_degrees}°"])
        else:
            table.add_row(["Rotation Aug", "✗ Disabled"])
        print(table.get_string(title="Training Dataset"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))
        print()

        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))