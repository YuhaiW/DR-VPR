"""
Inference-latency benchmark for DR-VPR v2 (P1 standalone two-stage rerank).

Measures per-query latency components:
  (a) BoQ(ResNet50)@320 forward — produces stage-1 16384-d descriptor
  (b) E2ResNet(C8) multi-scale forward — produces stage-2 1024-d descriptor
  (c) FAISS top-100 search against a representative database (401 db images for
      ConSLAM, ~4000 for ConPR — measure both sizes)
  (d) Rerank score computation: (1-β)·boq_sim + β·equi_sim over top-100, argmax

Reports both "model-forward only" (apples-to-apples with single-stage baselines
that report only forward latency) and "end-to-end including retrieve + rerank".

All measurements: batch=1, 320×320 input, fp32, warmup 30, timed 100 iters.
RTX 5090, CUDA event-based timing for sub-ms accuracy.
"""
import os
import time
import numpy as np
import torch
import torch.nn.functional as F
import faiss

DEVICE = 'cuda:0'
WARMUP = 30
N_ITER = 100
IMG_SIZE = (320, 320)
BETA = 0.10
TOP_K = 100

# Representative db sizes
DB_CONSLAM = 401     # Sequence5 image count
DB_CONPR_AVG = 3500  # rough avg across 10 sequences (3017–4672)

P1_CKPT = 'LOGS/equi_standalone_seed1_ms/lightning_logs/version_0/checkpoints/equi_ms_seed1_epoch(07)_R1[0.3359].ckpt'


def load_official_boq():
    model = torch.hub.load(
        "amaralibey/bag-of-queries", "get_trained_boq",
        backbone_name="resnet50", output_dim=16384,
    )
    return model.eval().to(DEVICE)


def load_p1_equi(ckpt_path):
    from models.equi_multiscale import E2ResNetMultiScale
    model = E2ResNetMultiScale(
        orientation=8, layers=(2, 2, 2, 2),
        channels=(64, 128, 256, 512), out_dim=1024,
    )
    state = torch.load(ckpt_path, map_location='cpu')['state_dict']
    state = {k.replace('model.', '', 1) if k.startswith('model.') else k: v
             for k, v in state.items()}
    model.load_state_dict(state, strict=False)
    return model.to(DEVICE).eval()


@torch.no_grad()
def time_forward(model, x, label):
    """CUDA-event based timing of a single model forward, batch 1."""
    # warmup
    for _ in range(WARMUP):
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
    torch.cuda.synchronize()

    # Timed with CUDA events
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITER)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITER)]
    for i in range(N_ITER):
        starts[i].record()
        out = model(x)
        if isinstance(out, tuple):
            out = out[0]
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(f"  {label:36s}  mean={times_ms.mean():6.3f} ms  "
          f"std={times_ms.std():.3f}  min={times_ms.min():.3f}  "
          f"med={np.median(times_ms):.3f}")
    return times_ms


def time_two_stage_combined(boq, equi, x, label):
    """Time BoQ + E2ResNet sequentially (per-query forward sum)."""
    # warmup
    for _ in range(WARMUP):
        b = boq(x)
        if isinstance(b, tuple):
            b = b[0]
        e = equi(x)
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITER)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(N_ITER)]
    for i in range(N_ITER):
        starts[i].record()
        b = boq(x)
        if isinstance(b, tuple):
            b = b[0]
        b = F.normalize(b, p=2, dim=1)
        e = equi(x)  # already L2-normed in E2ResNetMultiScale.forward
        ends[i].record()
    torch.cuda.synchronize()
    times_ms = np.array([s.elapsed_time(e) for s, e in zip(starts, ends)])
    print(f"  {label:36s}  mean={times_ms.mean():6.3f} ms  "
          f"std={times_ms.std():.3f}  min={times_ms.min():.3f}  "
          f"med={np.median(times_ms):.3f}")
    return times_ms


def time_faiss_search(db_size, dim, top_k=100):
    """Time a single FAISS IndexFlatIP top-K query against a random db."""
    rng = np.random.RandomState(0)
    db = rng.randn(db_size, dim).astype(np.float32)
    db /= np.linalg.norm(db, axis=1, keepdims=True)
    q = rng.randn(1, dim).astype(np.float32)
    q /= np.linalg.norm(q, axis=1, keepdims=True)

    idx = faiss.IndexFlatIP(dim)
    idx.add(db)

    # warmup
    for _ in range(WARMUP):
        _ = idx.search(q, top_k)

    times_us = np.zeros(N_ITER)
    for i in range(N_ITER):
        t0 = time.perf_counter_ns()
        _, _ = idx.search(q, top_k)
        t1 = time.perf_counter_ns()
        times_us[i] = (t1 - t0) / 1000.0
    return times_us / 1000.0  # convert to ms


def time_rerank_score(top_k=100, equi_dim=1024, boq_dim=16384):
    """Time the rerank score computation: cosine sim + weighted argmax over top-K."""
    rng = np.random.RandomState(0)
    boq_q = rng.randn(boq_dim).astype(np.float32)
    boq_q /= np.linalg.norm(boq_q)
    boq_db_top = rng.randn(top_k, boq_dim).astype(np.float32)
    boq_db_top /= np.linalg.norm(boq_db_top, axis=1, keepdims=True)
    equi_q = rng.randn(equi_dim).astype(np.float32)
    equi_q /= np.linalg.norm(equi_q)
    equi_db_top = rng.randn(top_k, equi_dim).astype(np.float32)
    equi_db_top /= np.linalg.norm(equi_db_top, axis=1, keepdims=True)

    # warmup
    for _ in range(WARMUP):
        bs = boq_q @ boq_db_top.T
        es = equi_q @ equi_db_top.T
        score = (1 - BETA) * bs + BETA * es
        _ = int(np.argmax(score))

    times_us = np.zeros(N_ITER)
    for i in range(N_ITER):
        t0 = time.perf_counter_ns()
        bs = boq_q @ boq_db_top.T
        es = equi_q @ equi_db_top.T
        score = (1 - BETA) * bs + BETA * es
        _ = int(np.argmax(score))
        t1 = time.perf_counter_ns()
        times_us[i] = (t1 - t0) / 1000.0
    return times_us / 1000.0  # ms


def main():
    print("=" * 90)
    print("DR-VPR v2 (P1 standalone two-stage rerank) — inference latency benchmark")
    print("=" * 90)
    print(f"Hardware: {torch.cuda.get_device_name(0)}")
    print(f"Setting:  batch=1, 320×320, fp32, warmup={WARMUP}, timed={N_ITER}, β={BETA}, top-K={TOP_K}")
    print()

    print("[1/3] Loading models...")
    boq = load_official_boq()
    equi = load_p1_equi(P1_CKPT)
    boq_params = sum(p.numel() for p in boq.parameters())
    equi_params = sum(p.numel() for p in equi.parameters())
    total_params = boq_params + equi_params
    print(f"      BoQ(ResNet50):       {boq_params/1e6:6.2f} M params")
    print(f"      E2ResNet(C8) MS:     {equi_params/1e6:6.2f} M params")
    print(f"      DR-VPR v2 total:     {total_params/1e6:6.2f} M params")
    print()

    print("[2/3] Model-forward latency (CUDA-event timing)")
    print("-" * 90)
    x = torch.randn(1, 3, *IMG_SIZE, device=DEVICE)
    boq_times = time_forward(boq, x, '(a) BoQ forward (16384-d)')
    equi_times = time_forward(equi, x, '(b) E2ResNet forward (1024-d)')
    combined_times = time_two_stage_combined(boq, equi, x, '(a+b) Combined two-stage forward')
    print()

    print("[3/3] Retrieve + rerank latency (CPU)")
    print("-" * 90)
    faiss_conslam_times = time_faiss_search(DB_CONSLAM, 16384, TOP_K)
    print(f"  (c1) FAISS top-{TOP_K} search ConSLAM (db={DB_CONSLAM})  "
          f"mean={faiss_conslam_times.mean():6.3f} ms  med={np.median(faiss_conslam_times):.3f}")
    faiss_conpr_times = time_faiss_search(DB_CONPR_AVG, 16384, TOP_K)
    print(f"  (c2) FAISS top-{TOP_K} search ConPR avg (db={DB_CONPR_AVG})  "
          f"mean={faiss_conpr_times.mean():6.3f} ms  med={np.median(faiss_conpr_times):.3f}")
    rerank_times = time_rerank_score(TOP_K)
    print(f"  (d) Rerank score + argmax over top-{TOP_K}  "
          f"mean={rerank_times.mean():6.3f} ms  med={np.median(rerank_times):.3f}")
    print()

    print("=" * 90)
    print("END-TO-END LATENCY BREAKDOWN (per query)")
    print("=" * 90)
    print(f"{'Component':<48s}  {'mean (ms)':>10s}  {'median':>9s}")
    print("-" * 90)
    print(f"{'(a) BoQ forward':<48s}  {boq_times.mean():>10.3f}  {np.median(boq_times):>9.3f}")
    print(f"{'(b) E2ResNet multi-scale forward':<48s}  {equi_times.mean():>10.3f}  {np.median(equi_times):>9.3f}")
    print(f"{'(c1) FAISS top-100 search (ConSLAM db=401)':<48s}  {faiss_conslam_times.mean():>10.3f}  {np.median(faiss_conslam_times):>9.3f}")
    print(f"{'(c2) FAISS top-100 search (ConPR db=3500)':<48s}  {faiss_conpr_times.mean():>10.3f}  {np.median(faiss_conpr_times):>9.3f}")
    print(f"{'(d) Rerank score + argmax':<48s}  {rerank_times.mean():>10.3f}  {np.median(rerank_times):>9.3f}")
    print("-" * 90)
    e2e_conslam = boq_times.mean() + equi_times.mean() + faiss_conslam_times.mean() + rerank_times.mean()
    e2e_conpr   = boq_times.mean() + equi_times.mean() + faiss_conpr_times.mean()   + rerank_times.mean()
    model_only  = boq_times.mean() + equi_times.mean()
    print(f"{'>>> Model-only (a+b) [comparable to single-stage baselines]':<48s}  {model_only:>10.3f}")
    print(f"{'>>> End-to-end ConSLAM (a+b+c1+d)':<48s}  {e2e_conslam:>10.3f}")
    print(f"{'>>> End-to-end ConPR (a+b+c2+d)':<48s}  {e2e_conpr:>10.3f}")
    print()
    print(f"Throughput (model-only): {1000.0 / model_only:7.1f} FPS")
    print()
    print("Notes:")
    print("  - 'Model-only' is the apples-to-apples comparison vs single-stage")
    print("    baseline latencies (which also report just forward, not retrieve).")
    print("  - 'End-to-end' includes the FAISS search and rerank computation that")
    print("    are exclusive to two-stage methods. Single-stage baselines have")
    print("    just forward + FAISS top-1, which is similar to ours minus the")
    print("    rerank step (~0 ms) and minus the second forward pass.")
    print("  - FAISS search scales mildly with db size; numbers above are for")
    print("    representative ConSLAM (401) and ConPR-avg (3500) databases.")


if __name__ == '__main__':
    main()
