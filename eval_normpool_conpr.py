"""Quick ConPR full-10-seq eval for the L2-norm GroupPool checkpoint.

Loads the DualBranch C8 concat freeze_boq norm-pool checkpoint
(`resnet50_DualBranch_normpool_smoke_seed1` epoch 03), extracts BoQ
(Branch 1) and equi (Branch 2) descriptors at the matched 320x320
resolution, then runs ConPR full-10-seq R@1 at a range of β values to
fill in the missing entry in Table 3.

Mirrors the architecture build of eval_rerank.py exactly.
"""
import os
os.environ.setdefault('FUSION_METHOD', 'concat')
os.environ['GROUP_POOL_MODE'] = 'norm'

import glob
import sys
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from torch.utils.data import DataLoader

from conpr_eval_dataset_rot import InferDataset as ConPRDS, get_yaw_from_pose
from train_fusion import VPRModel, load_boq_pretrained


DEVICE = 'cuda:0'
BATCH_SIZE = 16
NUM_WORKERS = 4
IMG_SIZE = (320, 320)
TOP_K = 100
GT_THRES = 5.0
YAW_THRESHOLD = 80.0
BETA_VALUES = [0.0, 0.05, 0.10, 0.15, 0.20]

CONPR_PATH = './datasets/ConPR/'
DB_SEQ = '20230623'
Q_SEQS = ['20230531', '20230611', '20230627', '20230628', '20230706',
          '20230717', '20230803', '20230809', '20230818']

CKPT_DIR = ('LOGS/resnet50_DualBranch_normpool_smoke_seed1/'
            'lightning_logs/version_0/checkpoints')


def find_val_best_ckpt():
    paths = glob.glob(os.path.join(CKPT_DIR, '*.ckpt'))

    def r1(p):
        try:
            return float(p.split('R1[')[1].split(']')[0])
        except Exception:
            return -1.0
    return max(paths, key=r1)


def build_model():
    return VPRModel(
        backbone_arch='resnet50', pretrained=True,
        layers_to_freeze=2, layers_to_crop=[4],
        agg_arch='boq',
        agg_config={'in_channels': 1024, 'proj_channels': 512,
                     'num_queries': 64, 'num_layers': 2, 'row_dim': 32},
        use_dual_branch=True, equi_orientation=8,
        equi_layers=[2, 2, 2, 2], equi_channels=[64, 128, 256, 512],
        equi_out_dim=1024, fusion_method='concat', use_projection=False,
        lr=1e-3, optimizer='adamw', weight_decay=1e-4, momentum=0.9,
        warmpup_steps=300, milestones=[8, 14], lr_mult=0.3,
        loss_name='MultiSimilarityLoss', miner_name='MultiSimilarityMiner',
        miner_margin=0.1, faiss_gpu=False,
    )


@torch.no_grad()
def extract_descriptors(model, dl):
    descs1, descs2 = [], []
    for imgs, _ in dl:
        imgs = imgs.to(DEVICE)
        feat1 = model.backbone(imgs)
        feat2 = model.backbone2(imgs)
        d1 = model.aggregator.branch1_aggregator(feat1)
        d2 = model.aggregator.branch2_aggregator(feat2)
        d1 = F.normalize(d1, p=2, dim=1)
        d2 = F.normalize(d2, p=2, dim=1)
        descs1.append(d1.cpu().numpy())
        descs2.append(d2.cpu().numpy())
    return np.vstack(descs1), np.vstack(descs2)


def compute_recall(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                    db_poses, q_poses, beta):
    db_x, db_y = db_poses[:, 3], db_poses[:, 7]
    idx = faiss.IndexFlatIP(d_boq_db.shape[1])
    idx.add(d_boq_db)
    _, top_k = idx.search(d_boq_q, TOP_K)
    tp1 = total = 0
    for qi in range(len(q_poses)):
        ds = (q_poses[qi, 3] - db_x) ** 2 + (q_poses[qi, 7] - db_y) ** 2
        pp = np.where(ds < GT_THRES ** 2)[0]
        if len(pp) == 0:
            continue
        q_yaw = get_yaw_from_pose(q_poses[qi])
        ypos = set()
        for p in pp:
            d = abs(q_yaw - get_yaw_from_pose(db_poses[p]))
            if d > 180:
                d = 360 - d
            if d <= YAW_THRESHOLD:
                ypos.add(int(p))
        if not ypos:
            continue
        cands = top_k[qi]
        bs = d_boq_q[qi] @ d_boq_db[cands].T
        es = d_eq_q[qi] @ d_eq_db[cands].T
        score = (1 - beta) * bs + beta * es
        top1 = int(cands[np.argmax(score)])
        if top1 in ypos:
            tp1 += 1
        total += 1
    return tp1 / max(total, 1) * 100, total


def main():
    ckpt = find_val_best_ckpt()
    print(f'Checkpoint: {ckpt}')

    model = build_model()
    print('Loading BoQ pretrained weights...')
    load_boq_pretrained(model)
    state = torch.load(ckpt, map_location='cpu')['state_dict']
    missing, unexpected = model.load_state_dict(state, strict=False)
    real_missing = [k for k in missing if not k.endswith('.filter')]
    print(f'  load_state_dict: {len(missing)} missing '
          f'({len(real_missing)} real), {len(unexpected)} unexpected')
    model = model.to(DEVICE).eval()

    print(f'\nExtracting database descriptors ({DB_SEQ})...')
    db_ds = ConPRDS(DB_SEQ, dataset_path=CONPR_PATH, img_size=IMG_SIZE)
    db_dl = DataLoader(db_ds, batch_size=BATCH_SIZE, shuffle=False,
                        num_workers=NUM_WORKERS, pin_memory=True)
    d_boq_db, d_eq_db = extract_descriptors(model, db_dl)
    print(f'  db: BoQ={d_boq_db.shape}  equi={d_eq_db.shape}')

    per_seq_r1 = {b: [] for b in BETA_VALUES}
    print('\n--- Per-query-sequence R@1 ---')
    print(f'{"q-seq":>10s}  ' +
          '  '.join(f'β={b:.2f}' for b in BETA_VALUES) + '  N')
    for q_seq in Q_SEQS:
        q_ds = ConPRDS(q_seq, dataset_path=CONPR_PATH, img_size=IMG_SIZE)
        q_dl = DataLoader(q_ds, batch_size=BATCH_SIZE, shuffle=False,
                           num_workers=NUM_WORKERS, pin_memory=True)
        d_boq_q, d_eq_q = extract_descriptors(model, q_dl)
        row = []
        n = 0
        for b in BETA_VALUES:
            r1, n = compute_recall(d_boq_db, d_boq_q, d_eq_db, d_eq_q,
                                    db_ds.poses, q_ds.poses, b)
            per_seq_r1[b].append(r1)
            row.append(f'{r1:5.2f}')
        print(f'{q_seq:>10s}  ' + '  '.join(row) + f'  {n}')

    print('\n--- Mean over 9 ConPR query sequences ---')
    for b in BETA_VALUES:
        m = np.mean(per_seq_r1[b])
        print(f'  β={b:.2f}:  R@1 = {m:.2f}%')


if __name__ == '__main__':
    main()
