import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms as T

from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from . import PittsburgDataset
from dataloaders.ConPRDataset import ConPRDataset

# 🔥 导入我们创建的custom transforms
from dataloaders.transforms import RandomResizedCrop, RandomPatch, RandomRotation90

from prettytable import PrettyTable

IMAGENET_MEAN_STD = {'mean': [0.485, 0.456, 0.406], 
                     'std': [0.229, 0.224, 0.225]}

VIT_MEAN_STD = {'mean': [0.5, 0.5, 0.5], 
                'std': [0.5, 0.5, 0.5]}

TRAIN_CITIES = [
    'Bangkok',
    'BuenosAires',
    'LosAngeles',
    'MexicoCity',
    'OSL',
    'Rome',
    'Barcelona',
    'Chicago',
    'Madrid',
    'Miami',
    'Phoenix',
    'TRT',
    'Boston',
    'Lisbon',
    'Medellin',
    'Minneapolis',
    'PRG',
    'WashingtonDC',
    'Brussels',
    'London',
    'Melbourne',
    'Osaka',
    'PRS',
]


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
                 
                 # 🔥🔥🔥 新增参数：数据增强配置
                 use_random_crop=True,        # 是否使用Random Resized Crop
                 crop_scale=(0.7, 1.0),       # Crop的scale范围
                 crop_ratio=(0.9, 1.1),       # Crop的aspect ratio范围
                 use_rotation=True,           # 是否使用90度旋转
                 use_patch_augment=True,      # 是否使用patch遮挡
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
        
        # 🔥 保存augmentation配置
        self.use_random_crop = use_random_crop
        self.crop_scale = crop_scale
        self.crop_ratio = crop_ratio
        self.use_rotation = use_rotation
        self.use_patch_augment = use_patch_augment
        
        self.save_hyperparameters()

        # 🔥🔥🔥 构建训练transforms（根据配置）
        self._build_transforms()

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

    def _build_transforms(self):
        """
        🔥 构建训练和验证的transforms
        根据配置选择不同的augmentation策略
        """
        train_transform_list = []
        
        # ===== 几何变换 =====
        if self.use_random_crop:
            # 🔥 使用Random Resized Crop（核心改进）
            print(f"🔥 Using Random Resized Crop:")
            print(f"   - Scale: {self.crop_scale}")
            print(f"   - Ratio: {self.crop_ratio}")
            train_transform_list.append(
                RandomResizedCrop(
                    size=self.image_size,
                    scale=self.crop_scale,
                    ratio=self.crop_ratio
                )
            )
        else:
            # 标准resize
            print(f"Using standard resize (no crop augmentation)")
            train_transform_list.append(
                T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR)
            )
        
        # 其他几何增强
        train_transform_list.extend([
            T.RandomHorizontalFlip(p=0.5),
        ])
        
        if self.use_rotation:
            # 🔥 90度旋转（适合BEV图像）
            print(f"🔥 Using Random 90° Rotation")
            train_transform_list.append(
                RandomRotation90(probability=0.3)
            )
        
        # 小角度旋转
        train_transform_list.append(
            T.RandomRotation(degrees=10)
        )
        
        # ===== 颜色变换 =====
        train_transform_list.extend([
            T.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1
            ),
        ])
        
        # ===== 转换为Tensor =====
        train_transform_list.append(T.ToTensor())
        
        # ===== Patch遮挡 =====
        if self.use_patch_augment:
            # 🔥 随机patch遮挡（模拟施工设备遮挡）
            print(f"🔥 Using Random Patch Occlusion")
            train_transform_list.append(
                RandomPatch(
                    n_patches=3,
                    patch_size=(30, 50),
                    probability=0.5
                )
            )
        
        # ===== 归一化 =====
        train_transform_list.append(
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        )
        
        # 组合所有transforms
        self.train_transform = T.Compose(train_transform_list)
        
        # ===== 验证transforms（无augmentation）=====
        self.valid_transform = T.Compose([
            T.Resize(self.image_size, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=self.mean_dataset, std=self.std_dataset)
        ])

    def setup(self, stage):
        if stage == 'fit':
            # load train dataloader with reload routine
            self.reload()

            # load validation sets (pitts_val, msls_val, ...etc)
            self.val_datasets = []
            for valid_set_name in self.val_set_names:
                if valid_set_name.lower() == 'pitts30k_test':
                    self.val_datasets.append(PittsburgDataset.get_whole_test_set(
                        input_transform=self.valid_transform))
                elif valid_set_name.lower() == 'pitts30k_val':
                    self.val_datasets.append(PittsburgDataset.get_whole_val_set(
                        input_transform=self.valid_transform))
                # elif valid_set_name.lower() == 'msls_val':
                #     self.val_datasets.append(MapillaryDataset.MSLS(
                #         input_transform=self.valid_transform))
                else:
                    print(
                        f'Validation set {valid_set_name} does not exist or has not been implemented yet')
                    raise NotImplementedError
            if self.show_data_stats:
                self.print_stats()

    def reload(self):
        self.train_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform)

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
        print()  # print a new line
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
        table.add_row(
            ["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(
            ["# of iterations", f"{self.train_dataset.__len__()//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        
        # 🔥 显示augmentation信息
        table.add_row(["Random Crop", f"{'✓ Enabled' if self.use_random_crop else '✗ Disabled'}"])
        if self.use_random_crop:
            table.add_row(["  - Scale", f"{self.crop_scale}"])
            table.add_row(["  - Ratio", f"{self.crop_ratio}"])
        table.add_row(["Random Rotation", f"{'✓ Enabled' if self.use_rotation else '✗ Disabled'}"])
        table.add_row(["Patch Occlusion", f"{'✓ Enabled' if self.use_patch_augment else '✗ Disabled'}"])
        
        print(table.get_string(title="Training config"))