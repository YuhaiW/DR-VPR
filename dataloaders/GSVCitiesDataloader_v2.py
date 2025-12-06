import pytorch_lightning as pl
from torch.utils.data import ConcatDataset, DataLoader
from torchvision import transforms as T

from dataloaders.GSVCitiesDataset import GSVCitiesDataset
from dataloaders.ConPRDataset import ConPRDataset
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
                 
                 # ===== ConPR Configuration (CORRECTED) =====
                 use_conpr=False,
                 conpr_sequences=None,  # e.g., ['20230623', '20230624']
                 conpr_anchor_distance=10.0,  # meters between anchor positions
                 conpr_place_distance=5.0,    # meters for same place matching
                 conpr_place_db_path=None,
                 conpr_rebuild_place_db=False,
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
        
        # ConPR parameters (CORRECTED)
        self.use_conpr = use_conpr
        self.conpr_sequences = conpr_sequences or []
        self.conpr_anchor_distance = conpr_anchor_distance
        self.conpr_place_distance = conpr_place_distance
        self.conpr_place_db_path = conpr_place_db_path
        self.conpr_rebuild_place_db = conpr_rebuild_place_db
        
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
            self.reload()

            # Load validation sets
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
        """Load training datasets (GSV + optionally ConPR)"""
        datasets = []
        
        # Always load GSV-Cities dataset
        self.gsv_dataset = GSVCitiesDataset(
            cities=self.cities,
            img_per_place=self.img_per_place,
            min_img_per_place=self.min_img_per_place,
            random_sample_from_each_place=self.random_sample_from_each_place,
            transform=self.train_transform)
        datasets.append(self.gsv_dataset)
        print(f'[INFO] Loaded GSV-Cities: {len(self.gsv_dataset)} places, '
              f'{self.gsv_dataset.total_nb_images} images')
        
        # Optionally load ConPR dataset with spatial-based place recognition
        if self.use_conpr and len(self.conpr_sequences) > 0:
            self.conpr_dataset = ConPRDataset(
                sequences=self.conpr_sequences,
                img_per_place=self.img_per_place,
                min_img_per_place=self.min_img_per_place,
                random_sample_from_each_place=self.random_sample_from_each_place,
                transform=self.train_transform,
                anchor_distance_threshold=self.conpr_anchor_distance,
                place_distance_threshold=self.conpr_place_distance,
                place_db_path=self.conpr_place_db_path,
                rebuild_place_db=self.conpr_rebuild_place_db,
            )
            datasets.append(self.conpr_dataset)
            
            # Print ConPR statistics
            print(f'[INFO] Loaded ConPR: {len(self.conpr_dataset)} places, '
                  f'{self.conpr_dataset.total_nb_images} images')
            
            # Print cross-trajectory statistics
            stats = self.conpr_dataset.get_place_statistics()
            print(f'[INFO] ConPR Cross-trajectory places: {stats["cross_trajectory_places"]} / {stats["total_places"]}')
        else:
            self.conpr_dataset = None
        
        # Combine datasets
        if len(datasets) == 1:
            self.train_dataset = datasets[0]
        else:
            self.train_dataset = ConcatDataset(datasets)
            print(f'[INFO] Combined dataset: {len(self.train_dataset)} total places')

    def train_dataloader(self):
        self.reload()
        return DataLoader(dataset=self.train_dataset, **self.train_loader_config)

    def val_dataloader(self):
        val_dataloaders = []
        for val_dataset in self.val_datasets:
            val_dataloaders.append(DataLoader(
                dataset=val_dataset, **self.valid_loader_config))
        return val_dataloaders
    
    def add_new_conpr_trajectory(self, new_sequence):
        """Add a new ConPR trajectory to the existing dataset"""
        if not self.use_conpr or self.conpr_dataset is None:
            print("[ERROR] ConPR is not enabled or dataset not initialized")
            return
        
        print(f"\n[DataModule] Adding new ConPR trajectory: {new_sequence}")
        self.conpr_dataset.add_new_trajectory(new_sequence)
        
        # Update combined dataset
        datasets = [self.gsv_dataset, self.conpr_dataset]
        self.train_dataset = ConcatDataset(datasets)
        
        print(f'[DataModule] Updated combined dataset: {len(self.train_dataset)} total places')

    def print_stats(self):
        print()
        
        # ===== Training Dataset Statistics =====
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        
        # GSV-Cities stats
        table.add_row(["GSV-Cities", ""])
        table.add_row(["  # of cities", f"{len(self.cities)}"])
        table.add_row(["  # of places", f'{len(self.gsv_dataset)}'])
        table.add_row(["  # of images", f'{self.gsv_dataset.total_nb_images}'])
        
        # ConPR stats (if enabled)
        if self.use_conpr and self.conpr_dataset is not None:
            table.add_row(["ConPR", ""])
            table.add_row(["  # of sequences", f"{len(self.conpr_sequences)}"])
            table.add_row(["  # of places", f'{len(self.conpr_dataset)}'])
            table.add_row(["  # of images", f'{self.conpr_dataset.total_nb_images}'])
            
            # Get cross-trajectory statistics
            stats = self.conpr_dataset.get_place_statistics()
            table.add_row(["  Total anchor places", f'{stats["total_places"]}'])
            table.add_row(["  Cross-trajectory places", f'{stats["cross_trajectory_places"]}'])
            table.add_row(["  Single-trajectory places", f'{stats["single_trajectory_places"]}'])
            table.add_row(["  Anchor distance", f'{self.conpr_anchor_distance}m'])
            table.add_row(["  Place distance", f'{self.conpr_place_distance}m'])
        
        # Total stats
        table.add_row(["", ""])
        table.add_row(["Total # of places", f'{len(self.train_dataset)}'])
        if self.use_conpr and self.conpr_dataset is not None:
            total_images = self.gsv_dataset.total_nb_images + self.conpr_dataset.total_nb_images
            table.add_row(["Total # of images", f'{total_images}'])
        else:
            table.add_row(["Total # of images", f'{self.gsv_dataset.total_nb_images}'])
        
        print(table.get_string(title="Training Dataset"))
        print()

        # ===== Validation Datasets =====
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        for i, val_set_name in enumerate(self.val_set_names):
            table.add_row([f"Validation set {i+1}", f"{val_set_name}"])
        print(table.get_string(title="Validation Datasets"))
        print()

        # ===== Training Configuration =====
        table = PrettyTable()
        table.field_names = ['Data', 'Value']
        table.align['Data'] = "l"
        table.align['Value'] = "l"
        table.header = False
        table.add_row(["Batch size (PxK)", f"{self.batch_size}x{self.img_per_place}"])
        table.add_row(["# of iterations", f"{len(self.train_dataset)//self.batch_size}"])
        table.add_row(["Image size", f"{self.image_size}"])
        print(table.get_string(title="Training config"))
