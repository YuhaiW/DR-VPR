"""
Dual-Branch VPR Training Script
Combines standard ResNet + MixVPR with E2ResNet for rotation robustness
"""
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import utils

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models import helper_1 as helper


class VPRModel(pl.LightningModule):
    """
    Dual-branch VPR Model with Pytorch Lightning
    Branch 1: Standard CNN (ResNet + MixVPR)
    Branch 2: Equivariant CNN (E2ResNet + GeM)
    """

    def __init__(self,
                #---- Branch 1: Standard CNN
                backbone_arch='resnet50',
                pretrained=True,
                layers_to_freeze=1,
                layers_to_crop=[],
                
                #---- Branch 1: Aggregator
                agg_arch='MixVPR',
                agg_config={},
                
                #---- Branch 2: Equivariant CNN (NEW!)
                use_dual_branch=False,
                equi_orientation=4,
                equi_layers=[2, 2, 2, 2],
                equi_channels=[64, 128, 256, 512],
                equi_out_dim=512,
                fusion_method='attention',
                use_projection=False,
                
                #---- Train hyperparameters
                lr=0.03, 
                optimizer='sgd',
                weight_decay=1e-3,
                momentum=0.9,
                warmpup_steps=500,
                milestones=[5, 10, 15],
                lr_mult=0.3,
                
                #----- Loss
                loss_name='MultiSimilarityLoss', 
                miner_name='MultiSimilarityMiner', 
                miner_margin=0.1,
                faiss_gpu=False
                 ):
        super().__init__()
        
        # Store hyperparameters
        self.encoder_arch = backbone_arch
        self.pretrained = pretrained
        self.layers_to_freeze = layers_to_freeze
        self.layers_to_crop = layers_to_crop

        self.agg_arch = agg_arch
        self.agg_config = agg_config
        
        self.use_dual_branch = use_dual_branch
        self.equi_orientation = equi_orientation
        self.equi_out_dim = equi_out_dim
        self.fusion_method = fusion_method

        self.lr = lr
        self.optimizer = optimizer
        self.weight_decay = weight_decay
        self.momentum = momentum
        self.warmpup_steps = warmpup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult

        self.loss_name = loss_name
        self.miner_name = miner_name
        self.miner_margin = miner_margin
        
        self.save_hyperparameters()
        
        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

        self.faiss_gpu = faiss_gpu
        
        # ----------------------------------
        # Build architecture
        # ----------------------------------
        if not use_dual_branch:
            # Single branch (original)
            self.backbone = helper.get_backbone(
                backbone_arch, pretrained, layers_to_freeze, layers_to_crop
            )
            self.aggregator = helper.get_aggregator(agg_arch, agg_config)
            self.backbone2 = None
            
        else:
            # Dual branch
            print(f" Using Dual-Branch Architecture!")
            print(f"   Branch 1: {backbone_arch} + {agg_arch}")
            print(f"   Branch 2: E2ResNet (C{equi_orientation}) + GeM")
            print(f"   Fusion: {fusion_method}")
            
            # Branch 1: Standard CNN
            self.backbone = helper.get_backbone(
                backbone_arch, pretrained, layers_to_freeze, layers_to_crop
            )
            branch1_agg = helper.get_aggregator(agg_arch, agg_config)
            branch1_out_dim = agg_config.get('out_rows', 4) * agg_config['out_channels']
            
            # Branch 2: Equivariant CNN
            self.backbone2 = helper.get_equivariant_backbone(
                orientation=equi_orientation,
                layers=equi_layers,
                channels=equi_channels,
                pretrained=False
            )
            
            #  KEY FIX: Calculate correct output channels after GroupPooling
            # After GroupPooling: channels = equi_channels[-1] / orientation
            # Example: 512 / 8 = 64 channels
            branch2_in_channels = equi_channels[-1] // equi_orientation
            
            print(f"   Branch 2 output: {equi_channels[-1]} equivariant → {branch2_in_channels} invariant channels")
            print(f"   Branch 2 descriptor: {branch2_in_channels} → {equi_out_dim} dim")
            
            # Dual-branch aggregator
            self.aggregator = helper.get_dual_branch_aggregator(
                branch1_agg=branch1_agg,
                branch1_out_dim=branch1_out_dim,
                branch2_in_channels=branch2_in_channels,  # Use corrected channels
                branch2_out_dim=equi_out_dim,
                fusion_method=fusion_method,
                use_projection=use_projection
            )
            
            print(f"   Final descriptor: {branch1_out_dim} + {equi_out_dim} = {branch1_out_dim + equi_out_dim} dim")
        
    # def forward(self, x):
    #     """Forward pass with dual-branch support"""
    #     if not self.use_dual_branch:
    #         # Single branch
    #         x = self.backbone(x)
    #         x = self.aggregator(x)
    #     else:
    #         # Dual branch
    #         x1 = self.backbone(x)
    #         x2 = self.backbone2(x)
    #         x = self.aggregator(x1, x2)
        
    #     return x
    def forward(self, x, return_features=False):
        """Forward pass with dual-branch support"""
        if not self.use_dual_branch:
            # Single branch
            features = self.backbone(x)
            descriptor = self.aggregator(features)
            
            if return_features:
                return {
                    'descriptor': descriptor,
                    'branch1_features': features,
                    'branch1_descriptor': descriptor
                }
            return descriptor
        
        else:
            # Dual branch
            branch1_features = self.backbone(x)
            branch2_features = self.backbone2(x)
            
                # Normal forward
            descriptor = self.aggregator(branch1_features, branch2_features)
            return descriptor
    
    def configure_optimizers(self):
        if self.optimizer.lower() == 'sgd':
            optimizer = torch.optim.SGD(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay, 
                                        momentum=self.momentum)
        elif self.optimizer.lower() == 'adamw':
            optimizer = torch.optim.AdamW(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        elif self.optimizer.lower() == 'adam':
            optimizer = torch.optim.Adam(self.parameters(), 
                                        lr=self.lr, 
                                        weight_decay=self.weight_decay)
        else:
            raise ValueError(f'Optimizer {self.optimizer} has not been added')
        
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_mult)
        return [optimizer], [scheduler]
    
    def optimizer_step(self, epoch, batch_idx,
                        optimizer, optimizer_idx, optimizer_closure,
                        on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # Warm up lr
        if self.trainer.global_step < self.warmpup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmpup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.lr
        optimizer.step(closure=optimizer_closure)
        
    def loss_function(self, descriptors, labels):
        # Mine pairs/triplets if there is an online mining strategy
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            
            # Calculate % of trivial pairs/triplets
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined/nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if type(loss) == tuple:
                loss, batch_acc = loss

        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc), 
                 prog_bar=True, logger=True)
        return loss
    
    def training_step(self, batch, batch_idx):
        places, labels = batch
        
        BS, N, ch, h, w = places.shape
        
        # Reshape
        images = places.view(BS*N, ch, h, w)
        labels = labels.view(-1)

        # Forward
        descriptors = self(images)
        loss = self.loss_function(descriptors, labels)
        
        self.log('loss', loss.item(), logger=True)
        return {'loss': loss}
    
    def training_epoch_end(self, training_step_outputs):
        self.batch_acc = []

    def validation_step(self, batch, batch_idx, dataloader_idx=None):
        places, _ = batch
        descriptors = self(places)
        return descriptors.detach().cpu()
    
    def validation_epoch_end(self, val_step_outputs):
        dm = self.trainer.datamodule
        
        # Handle single validation set
        if len(dm.val_datasets) == 1:
            val_step_outputs = [val_step_outputs]
        
        for i, (val_set_name, val_dataset) in enumerate(zip(dm.val_set_names, dm.val_datasets)):
            feats = torch.concat(val_step_outputs[i], dim=0)
            
            if 'pitts' in val_set_name:
                num_references = val_dataset.dbStruct.numDb
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.getPositives()
            elif 'msls' in val_set_name:
                num_references = val_dataset.num_references
                num_queries = len(val_dataset) - num_references
                positives = val_dataset.pIdx
            elif 'conpr' in val_set_name.lower():
                num_references = val_dataset.num_db
                num_queries = val_dataset.num_queries
                positives = val_dataset.getPositives()
            else:
                print(f'Please implement validation_epoch_end for {val_set_name}')
                raise NotImplementedError

            r_list = feats[:num_references]
            q_list = feats[num_references:]
            
            pitts_dict = utils.get_validation_recalls(
                r_list=r_list, 
                q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives,
                print_results=True,
                dataset_name=val_set_name,
                faiss_gpu=self.faiss_gpu
            )
            
            del r_list, q_list, feats, num_references, positives

            self.log(f'{val_set_name}/R1', pitts_dict[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', pitts_dict[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', pitts_dict[10], prog_bar=False, logger=True)
        
        print('\n\n')


if __name__ == '__main__':
    pl.utilities.seed.seed_everything(seed=190223, workers=True)
    
    # Datamodule
    datamodule = GSVCitiesDataModule(
        batch_size=60,  # Reduced for dual-branch
        img_per_place=4,
        min_img_per_place=4,
        shuffle_all=False,
        random_sample_from_each_place=True,
        image_size=(320, 320),
        num_workers=28,
        show_data_stats=True,
        val_set_names=['pitts30k_val', 'pitts30k_test', 'conpr'],
                # Or full validation: all 10 sequences (will be slower)
        conpr_sequences=None,
        
        conpr_yaw_threshold=80.0,
    )
    
    # ========================================
    # Dual-Branch Model Configuration
    # ========================================
    
    # Branch 2 configuration
    equi_orientation = 8
    equi_channels = [64, 128, 256, 512]
    
    # Calculate actual output channels after GroupPooling
    # GroupPooling reduces channels by the group order
    branch2_pooled_channels = equi_channels[-1] // equi_orientation  # 512 / 8 = 64
    
    print(f"\n{'='*60}")
    print(f"Dual-Branch VPR Configuration:")
    print(f"  Branch 1: ResNet50 + MixVPR")
    print(f"    - Output: 4 × 1024 = 4096 dim")
    print(f"  Branch 2: E2ResNet (C{equi_orientation}) + GeM")
    print(f"    - Equivariant channels: {equi_channels}")
    print(f"    - After GroupPooling: {branch2_pooled_channels} channels")
    print(f"    - Descriptor: 512 dim")
    print(f"  Final descriptor: 4096 + 512 = 4608 dim")
    print(f"{'='*60}\n")
    
    model = VPRModel(
        #---- Branch 1: Standard CNN
        backbone_arch='resnet50',
        pretrained=True,
        layers_to_freeze=2,
        layers_to_crop=[4],  # Crop layer4, use layer3 output (1024 channels)
        
        #---- Branch 1: Aggregator (MixVPR)
        agg_arch='MixVPR',
        agg_config={
            'in_channels': 1024,      # ResNet layer3 output
            'in_h': 20,               # 320 / 16 = 20
            'in_w': 20,
            'out_channels': 1024,
            'mix_depth': 4,
            'mlp_ratio': 1,
            'out_rows': 4
        },  # Output: 4 × 1024 = 4096 dim
        
        #---- Branch 2: Equivariant CNN
        use_dual_branch=True,              # Enable dual-branch
        equi_orientation=equi_orientation,  # C8 rotation group
        equi_layers=[2, 2, 2, 2],          # ResNet18-like
        equi_channels=equi_channels,       # [64, 128, 256, 512]
        equi_out_dim=512,                  # Final descriptor dimension
        fusion_method='attention', #'attention',            # Concatenate both branches
        use_projection=False,
        
        #---- Train hyperparameters
        lr=0.04,
        optimizer='sgd',
        weight_decay=0.001,
        momentum=0.9,
        warmpup_steps=650,
        milestones=[5, 10, 15, 25],
        lr_mult=0.3,
        
        #---- Loss
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False
    )
    
    # Checkpoint callback
    checkpoint_cb = ModelCheckpoint(
        monitor='pitts30k_val/R1',
        filename=f'{model.encoder_arch}_DualBranch_C{model.equi_orientation}' +
        '_epoch({epoch:02d})_R1[{pitts30k_val/R1:.4f}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=3,
        mode='max',
    )
    
    # Trainer
    trainer = pl.Trainer(
        accelerator='gpu', 
        devices=[0],
        default_root_dir=f'./LOGS/{model.encoder_arch}_DualBranch',
        num_sanity_val_steps=0,
        precision=16,  # Mixed precision training
        max_epochs=40,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )
    
    # Start training
    print("\n" + "="*60)
    print("Starting Dual-Branch VPR Training")
    print("  - Model: ResNet50 + MixVPR || E2ResNet + GeM")
    print("  - Descriptor: 4608 dim (4096 + 512)")
    print("  - Dataset: GSV-Cities")
    print("  - Validation: Pittsburgh 30k")
    print("="*60 + "\n")
    
    trainer.fit(model=model, datamodule=datamodule)
