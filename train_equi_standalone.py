"""
Standalone E2ResNet multi-scale training for DR-VPR rerank stage-2.

Trains ONLY the equivariant branch — no BoQ, no fusion, no gate. Loss is vanilla
MultiSimilarityLoss on the L2-normalized 1024-d desc_equi.

Use eval_rerank_standalone.py after training to do two-stage rerank with the
official BoQ(ResNet50) frozen as stage-1.
"""
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.optim import lr_scheduler
import utils

from dataloaders.GSVCitiesDataloader import GSVCitiesDataModule
from models.equi_multiscale import E2ResNetMultiScale


class EquiStandaloneVPR(pl.LightningModule):
    """Single-branch multi-scale equivariant VPR model.

    Only the equi branch exists — no BoQ. Trained with MS loss directly.
    """

    def __init__(
        self,
        orientation=8,
        layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512),
        out_dim=1024,
        gem_p_init=3.0,
        lr=1e-3,
        weight_decay=1e-4,
        warmup_steps=300,
        milestones=(8, 14),
        lr_mult=0.3,
        loss_name='MultiSimilarityLoss',
        miner_name='MultiSimilarityMiner',
        miner_margin=0.1,
        faiss_gpu=False,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.milestones = milestones
        self.lr_mult = lr_mult
        self.faiss_gpu = faiss_gpu

        self.model = E2ResNetMultiScale(
            orientation=orientation, layers=layers, channels=channels,
            out_dim=out_dim, gem_p_init=gem_p_init,
        )

        self.loss_fn = utils.get_loss(loss_name)
        self.miner = utils.get_miner(miner_name, miner_margin)
        self.batch_acc = []

    def forward(self, x):
        return self.model(x)   # already L2-normalized 1024-d

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr,
                                weight_decay=self.weight_decay)
        for pg in opt.param_groups:
            pg['initial_lr'] = pg['lr']
        sched = lr_scheduler.MultiStepLR(opt, milestones=list(self.milestones),
                                          gamma=self.lr_mult)
        return [opt], [sched]

    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx,
                       optimizer_closure, on_tpu=False, using_native_amp=False,
                       using_lbfgs=False):
        if self.trainer.global_step < self.warmup_steps:
            lr_scale = min(1., float(self.trainer.global_step + 1) / self.warmup_steps)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * pg.get('initial_lr', self.lr)
        optimizer.step(closure=optimizer_closure)

    def loss_function(self, descriptors, labels):
        if self.miner is not None:
            miner_outputs = self.miner(descriptors, labels)
            loss = self.loss_fn(descriptors, labels, miner_outputs)
            nb_samples = descriptors.shape[0]
            nb_mined = len(set(miner_outputs[0].detach().cpu().numpy()))
            batch_acc = 1.0 - (nb_mined / nb_samples)
        else:
            loss = self.loss_fn(descriptors, labels)
            batch_acc = 0.0
            if isinstance(loss, tuple):
                loss, batch_acc = loss
        self.batch_acc.append(batch_acc)
        self.log('b_acc', sum(self.batch_acc) / len(self.batch_acc),
                 prog_bar=True, logger=True)
        return loss

    def training_step(self, batch, batch_idx):
        places, labels = batch
        BS, N, ch, h, w = places.shape
        images = places.view(BS * N, ch, h, w)
        labels = labels.view(-1)
        descriptors = self(images)            # already L2-normalized
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
            elif 'conslam' in val_set_name.lower():
                num_references = val_dataset.num_db
                num_queries = val_dataset.num_queries
                positives = val_dataset.getPositives()
            else:
                raise NotImplementedError(f'val for {val_set_name} not implemented')

            r_list = feats[:num_references]
            q_list = feats[num_references:]

            d = utils.get_validation_recalls(
                r_list=r_list, q_list=q_list,
                k_values=[1, 5, 10, 15, 20, 50, 100],
                gt=positives, print_results=True,
                dataset_name=val_set_name, faiss_gpu=self.faiss_gpu,
            )
            del r_list, q_list, feats, num_references, positives
            self.log(f'{val_set_name}/R1', d[1], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R5', d[5], prog_bar=False, logger=True)
            self.log(f'{val_set_name}/R10', d[10], prog_bar=False, logger=True)
        print('\n\n')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=190223)
    parser.add_argument('--max_epochs', type=int, default=10)
    args = parser.parse_args()

    pl.utilities.seed.seed_everything(seed=args.seed, workers=True)
    print(f"Seed: {args.seed}")

    datamodule = GSVCitiesDataModule(
        batch_size=32, img_per_place=4, min_img_per_place=4,
        shuffle_all=False, random_sample_from_each_place=True,
        image_size=(320, 320), num_workers=12, show_data_stats=True,
        val_set_names=['conpr', 'conslam'],   # both monitored
        conpr_sequences=None, conpr_yaw_threshold=80.0,
    )

    print(f"\n{'='*60}")
    print("Standalone E2ResNet Multi-scale Training (P1)")
    print("  Single branch: E2ResNet C8 + multi-scale GroupPool + GeM + Linear")
    print("  No BoQ, no fusion. Pure equi for stage-2 rerank.")
    print(f"{'='*60}\n")

    model = EquiStandaloneVPR(
        orientation=8, layers=(2, 2, 2, 2), channels=(64, 128, 256, 512),
        out_dim=1024, gem_p_init=3.0,
        lr=1e-3, weight_decay=1e-4, warmup_steps=300,
        milestones=(8, 14), lr_mult=0.3,
        loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner',
        miner_margin=0.1, faiss_gpu=False,
    )

    n_params = sum(p.numel() for p in model.parameters())
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: {n_params:,} total, {n_train:,} trainable ({n_train/n_params*100:.1f}%)")

    _tag = os.environ.get('RUN_TAG', 'equi_standalone_ms')
    run_tag = f"equi_standalone_seed{args.seed}_{_tag}"
    checkpoint_cb = ModelCheckpoint(
        monitor='conslam/R1',
        filename=f'equi_ms_seed{args.seed}_epoch({{epoch:02d}})_R1[{{conslam/R1:.4f}}]',
        auto_insert_metric_name=False,
        save_weights_only=True,
        save_top_k=-1,
        mode='max',
    )

    trainer = pl.Trainer(
        accelerator='gpu', devices=[0],
        default_root_dir=f'./LOGS/{run_tag}',
        num_sanity_val_steps=0, precision=16,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1,
        callbacks=[checkpoint_cb],
        reload_dataloaders_every_n_epochs=1,
        log_every_n_steps=20,
    )

    print("\n" + "=" * 60)
    print("Starting standalone equi training")
    print(f"  out_dim: 1024, max_epochs: {args.max_epochs}")
    print(f"  Val: conpr (single-stage retrieve, R@1) + conslam (R@1, ckpt monitor)")
    print("=" * 60 + "\n")

    trainer.fit(model=model, datamodule=datamodule)
