"""
时序版GSVCitiesDataModule
替换原来的 dataloaders/GSVCitiesDataloader.py

改进：
- 向后兼容原有单帧模式
- 支持时序模式
"""
import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.temporal_gsv_dataset import GSVCitiesDataset  # 使用修改后的
from . import PittsburgDataset

from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok', 'BuenosAires', 'LosAngeles', 'MexicoCity', 'OSL', 'Rome',
    'Barcelona', 'Chicago', 'Madrid', 'Miami', 'Phoenix', 'TRT',
    'Boston', 'Lisbon', 'Medellin', 'Minneapolis', 'PRG', 'WashingtonDC',
    'Brussels', 'London', 'Melbourne', 'Osaka', 'PRS',
]


class GSVCitiesDataModule(pl.LightningDataModule):
    """
    时序版DataModule
    
    新参数：
        use_temporal: 是否使用时序模式
        temporal_length: 时序长度
        temporal_stride: 帧间间隔
        random_temporal: 训练时是否随机时序长度
        temporal_for_val: 验证时是否也使用时序
    """
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
                 # 时序参数
                 use_temporal=False,          # ← 新增：是否使用时序
                 temporal_length=5,           # ← 新增：时序长度
                 temporal_stride=1,           # ← 新增：帧间间隔
                 random_temporal=True,        # ← 新增：随机时序
                 temporal_for_val=False,      # ← 新增：验证时是否时序
                 ):
        super().__init__()
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
        
        # 时序配置
        self.use_temporal = use_temporal
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.random_temporal = random_temporal
        self.temporal_for_val = temporal_for_val
        
        self.save_hyperparameters()

        self.train_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.RandAugment(num_ops=3, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset),
        ])

        self.valid_transform = T.Compose([
            T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)])

        self.train_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': self.shuffle_all}

        self.valid_loader_config = {
            'batch_size': self.batch_size,
            'num_workers': self.num_workers//2,
            'drop_last': False,
            'pin_memory': True,
            'shuffle': False}

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # load validation sets
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == 'pitts30k_test':
                    self.val_datasets.append(PittsburgDataset.get_whole_test_set(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'pitts30k_val':
                    self.val_datasets.append(PittsburgDataset.get_whole_val_set(
                        input_transform=self.valid_transform))
                else:
                    print(f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
                    
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        """创建训练数据集"""
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform,
            # 时序参数
            temporal_length=self.temporal_length if self.use_temporal else 1,
            temporal_stride=self.temporal_stride,
            random_temporal=self.random_temporal if self.use_temporal else False,
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
        
        # 时序信息
        if self.use_temporal:
            table.add_row(["Temporal mode", "✓ Enabled"])
            table.add_row(["Temporal length", f"{self.temporal_length}"])
            table.add_row(["Temporal stride", f"{self.temporal_stride}"])
            table.add_row(["Random temporal", f"{'✓' if self.random_temporal else '✗'}"])
        else:
            table.add_row(["Temporal mode", "✗ Disabled (Single frame)"])
        
        print(table.get_string(title="Training config"))
