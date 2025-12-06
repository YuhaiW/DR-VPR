"""
时序版GSVCitiesDataset
替换原来的 dataloaders/GSVCitiesDataset.py

改进：
- 向后兼容原有单帧模式
- 支持时序模式（temporal_length > 1）
- 自动处理边界情况
"""
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import numpy as np

default_transform = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# NOTE: Hard coded path to dataset folder 
BASE_PATH = './datasets/GSV-Cities/'

if not Path(BASE_PATH).exists():
    raise FileNotFoundError(
        'BASE_PATH is hardcoded, please adjust to point to gsv_cities')


class GSVCitiesDataset(Dataset):
    """
    时序版GSV-Cities数据集
    
    新参数：
        temporal_length: 每个样本返回的帧数（默认1=单帧模式）
        temporal_stride: 帧间间隔
        random_temporal: 训练时是否随机时序长度
    """
    def __init__(self,
                 cities=['London', 'Boston'],
                 img_per_place=4,
                 min_img_per_place=4,
                 random_sample_from_each_place=True,
                 transform=default_transform,
                 base_path=BASE_PATH,
                 # 时序参数
                 temporal_length=1,          # ← 新增：时序长度
                 temporal_stride=1,          # ← 新增：帧间间隔
                 random_temporal=False,      # ← 新增：随机时序
                 ):
        super(GSVCitiesDataset, self).__init__()
        self.base_path = base_path
        self.cities = cities

        assert img_per_place <= min_img_per_place, \
            f"img_per_place should be less than {min_img_per_place}"
        self.img_per_place = img_per_place
        self.min_img_per_place = min_img_per_place
        self.random_sample_from_each_place = random_sample_from_each_place
        self.transform = transform
        
        # 时序配置
        self.temporal_length = temporal_length
        self.temporal_stride = temporal_stride
        self.random_temporal = random_temporal
        self.is_temporal = temporal_length > 1
        
        # generate the dataframe contraining images metadata
        self.dataframe = self.__getdataframes()
        
        # ═══════════════════════════════════════════
        # 新增：构建每个place的图片列表（用于时序采样）
        # ═══════════════════════════════════════════
        if self.is_temporal:
            self._build_place_image_lists()
        
        # get all unique place ids
        self.places_ids = pd.unique(self.dataframe.index)
        self.total_nb_images = len(self.dataframe)
        
    def __getdataframes(self):
        ''' 
            Return one dataframe containing
            all info about the images from all cities
        '''
        # read the first city dataframe
        df = pd.read_csv(self.base_path+'Dataframes/'+f'{self.cities[0]}.csv')
        df = df.sample(frac=1)  # shuffle the city dataframe
        
        # append other cities one by one
        for i in range(1, len(self.cities)):
            tmp_df = pd.read_csv(
                self.base_path+'Dataframes/'+f'{self.cities[i]}.csv')
            prefix = i
            tmp_df['place_id'] = tmp_df['place_id'] + (prefix * 10**5)
            tmp_df = tmp_df.sample(frac=1)  # shuffle the city dataframe
            df = pd.concat([df, tmp_df], ignore_index=True)

        # keep only places depicted by at least min_img_per_place images
        res = df[df.groupby('place_id')['place_id'].transform(
            'size') >= self.min_img_per_place]
        return res.set_index('place_id')
    
    def _build_place_image_lists(self):
        """构建每个place的图片列表（按时间排序）"""
        self.place_image_lists = {}
        
        for place_id in pd.unique(self.dataframe.index):
            place_df = self.dataframe.loc[place_id]
            
            # 如果只有一行，转为DataFrame
            if isinstance(place_df, pd.Series):
                place_df = place_df.to_frame().T
            
            # 按时间排序（year, month）
            place_df_sorted = place_df.sort_values(
                by=['year', 'month', 'lat'], ascending=True)
            
            # 保存排序后的索引和信息
            self.place_image_lists[place_id] = place_df_sorted
    
    def __getitem__(self, index):
        place_id = self.places_ids[index]
        
        # ═══════════════════════════════════════════
        # 单帧模式（原有逻辑）
        # ═══════════════════════════════════════════
        if not self.is_temporal:
            return self._get_single_frame(place_id)
        
        # ═══════════════════════════════════════════
        # 时序模式（新逻辑）
        # ═══════════════════════════════════════════
        else:
            return self._get_temporal_frames(place_id)
    
    def _get_single_frame(self, place_id):
        """原有的单帧采样逻辑"""
        # get the place in form of a dataframe
        place = self.dataframe.loc[place_id]
        
        # sample K images (rows) from this place
        if self.random_sample_from_each_place:
            if isinstance(place, pd.Series):
                place = place.to_frame().T
            place = place.sample(n=self.img_per_place)
        else:  # always get the same most recent images
            if isinstance(place, pd.Series):
                place = place.to_frame().T
            place = place.sort_values(
                by=['year', 'month', 'lat'], ascending=False)
            place = place[: self.img_per_place]
            
        imgs = []
        for i, row in place.iterrows():
            img_name = self.get_img_name(row)
            img_path = self.base_path + 'Images/' + \
                row['city_id'] + '/' + img_name
            img = self.image_loader(img_path)

            if self.transform is not None:
                img = self.transform(img)

            imgs.append(img)

        return torch.stack(imgs), torch.tensor(place_id).repeat(self.img_per_place)
    
    def _get_temporal_frames(self, place_id):
        """新增的时序采样逻辑"""
        place_df = self.place_image_lists[place_id]
        
        # 确定时序长度
        if self.random_temporal:
            # 训练时随机1到temporal_length帧
            T = np.random.randint(1, self.temporal_length + 1)
        else:
            T = self.temporal_length
        
        # 采样img_per_place个时序样本
        all_temporal_sequences = []
        
        for _ in range(self.img_per_place):
            # 随机选择中心帧
            if self.random_sample_from_each_place:
                center_idx = np.random.randint(0, len(place_df))
            else:
                # 固定采样：选择最新的图片作为中心
                center_idx = len(place_df) - 1
            
            # 构建时序窗口
            temporal_sequence = []
            half_window = T // 2
            
            for offset in range(-half_window, half_window + 1):
                if len(temporal_sequence) >= T:
                    break
                    
                frame_idx = center_idx + offset * self.temporal_stride
                
                # 边界处理：重复边界帧
                if frame_idx < 0:
                    frame_idx = 0
                elif frame_idx >= len(place_df):
                    frame_idx = len(place_df) - 1
                
                # 加载图片
                row = place_df.iloc[frame_idx]
                img_name = self.get_img_name(row)
                img_path = self.base_path + 'Images/' + \
                    row['city_id'] + '/' + img_name
                img = self.image_loader(img_path)
                
                if self.transform is not None:
                    img = self.transform(img)
                
                temporal_sequence.append(img)
            
            # 确保恰好T帧（处理边界情况）
            while len(temporal_sequence) < T:
                temporal_sequence.append(temporal_sequence[-1])
            
            # Stack成 [T, C, H, W]
            temporal_sequence = torch.stack(temporal_sequence[:T])
            all_temporal_sequences.append(temporal_sequence)
        
        # Stack成 [K, T, C, H, W]
        all_temporal_sequences = torch.stack(all_temporal_sequences)
        
        # Label重复K次
        labels = torch.tensor(place_id).repeat(self.img_per_place)
        
        return all_temporal_sequences, labels

    def __len__(self):
        '''Denotes the total number of places (not images)'''
        return len(self.places_ids)

    @staticmethod
    def image_loader(path):
        return Image.open(path).convert('RGB')

    @staticmethod
    def get_img_name(row):
        # given a row from the dataframe
        # return the corresponding image name
        city = row['city_id']
        
        # now remove the two digit we added to the id
        pl_id = row.name % 10**5  
        pl_id = str(pl_id).zfill(7)
        
        panoid = row['panoid']
        year = str(row['year']).zfill(4)
        month = str(row['month']).zfill(2)
        northdeg = str(row['northdeg']).zfill(3)
        lat, lon = str(row['lat']), str(row['lon'])
        name = city+'_'+pl_id+'_'+year+'_'+month+'_' + \
            northdeg+'_'+lat+'_'+lon+'_'+panoid+'.jpg'
        return name


# ════════════════════════════════════════════════════════════
# 测试代码
# ════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("="*70)
    print("  测试时序GSVCitiesDataset")
    print("="*70)
    
    # 测试单帧模式（向后兼容）
    print("\n【测试1】单帧模式（原有逻辑）")
    print("-"*70)
    dataset_single = GSVCitiesDataset(
        cities=['London'],
        img_per_place=4,
        min_img_per_place=4,
        temporal_length=1,  # 单帧
    )
    
    places, labels = dataset_single[0]
    print(f"  输出shape: {places.shape}")  # [4, 3, 224, 224]
    print(f"  Labels: {labels.shape}")
    print(f"  数据集大小: {len(dataset_single)} places")
    
    # 测试时序模式
    print("\n【测试2】时序模式（新功能）")
    print("-"*70)
    dataset_temporal = GSVCitiesDataset(
        cities=['London'],
        img_per_place=4,
        min_img_per_place=4,
        temporal_length=5,  # 时序5帧
        temporal_stride=1,
        random_temporal=False,
    )
    
    places, labels = dataset_temporal[0]
    print(f"  输出shape: {places.shape}")  # [4, 5, 3, 224, 224]
    print(f"  Labels: {labels.shape}")
    print(f"  解释: [K={places.shape[0]}, T={places.shape[1]}, C, H, W]")
    
    # 测试随机时序
    print("\n【测试3】随机时序长度")
    print("-"*70)
    dataset_random_temporal = GSVCitiesDataset(
        cities=['London'],
        img_per_place=4,
        min_img_per_place=4,
        temporal_length=5,  
        random_temporal=True,  # 随机1-5帧
    )
    
    for i in range(3):
        places, labels = dataset_random_temporal[i]
        print(f"  样本{i}: shape={places.shape}, T可能是1-5之间随机")
    
    print("\n" + "="*70)
    print("  测试完成！数据集支持单帧和时序两种模式")
    print("="*70)
