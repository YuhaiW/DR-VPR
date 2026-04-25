# Per-INPLANE-ROTATION bucket decomposition of R@1

Buckets are 5° wide (with a single tail bucket "25°+").  
In-plane rotation = TRUE residual rotation about the optical axis between query and the nearest-yaw GT positive (robotics convention, X = optical axis).  
BoQ baseline = official BoQ(ResNet50)@320, β=0.  
DR-VPR = BoQ ⊕ C16 multi-scale equi rerank, β=0.10.


## CONSLAM

_Pooled across 3 seeds; total queries (with multiplicity) = 921._

### Buckets ordered by in-plane rotation

| Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 | flip→✓ | flip→✗ |
|---|---:|---:|---:|---:|---:|---:|
| [ 0°,  5°) | 435 | 66.21% | 65.98% | **-0.23** | 8 | 9 |
| [ 5°, 10°) | 402 | 56.72% | 60.20% | **+3.48** | 15 | 1 |
| [10°, 15°) | 60 | 75.00% | 75.00% | **+0.00** | 0 | 0 |
| [15°, 20°) | 18 | 16.67% | 16.67% | **+0.00** | 0 | 0 |
| [20°, 25°) | 6 | 0.00% | 0.00% | **+0.00** | 0 | 0 |
| 25°+ | 0 | — | — | — | — | — |
| TOTAL | 921 | 61.24% | 62.65% | **+1.41** | 23 | 10 |

### Buckets ranked by ΔR@1 (largest gain first)

| Rank | Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 |
|---:|---|---:|---:|---:|---:|
| 1 | [ 5°, 10°) | 402 | 56.72% | 60.20% | **+3.48** |
| 2 | [10°, 15°) | 60 | 75.00% | 75.00% | **+0.00** |
| 3 | [15°, 20°) | 18 | 16.67% | 16.67% | **+0.00** |
| 4 | [20°, 25°) | 6 | 0.00% | 0.00% | **+0.00** |
| 5 | [ 0°,  5°) | 435 | 66.21% | 65.98% | **-0.23** |

## CONPR

_Pooled across 3 seeds; total queries (with multiplicity) = 7401._

### Buckets ordered by in-plane rotation

| Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 | flip→✓ | flip→✗ |
|---|---:|---:|---:|---:|---:|---:|
| [ 0°,  5°) | 4662 | 80.76% | 81.60% | **+0.84** | 71 | 32 |
| [ 5°, 10°) | 1857 | 83.20% | 83.31% | **+0.11** | 10 | 8 |
| [10°, 15°) | 711 | 85.65% | 84.39% | **-1.27** | 0 | 9 |
| [15°, 20°) | 111 | 72.97% | 69.37% | **-3.60** | 0 | 4 |
| [20°, 25°) | 54 | 50.00% | 50.00% | **+0.00** | 0 | 0 |
| 25°+ | 6 | 100.00% | 100.00% | **+0.00** | 0 | 0 |
| TOTAL | 7401 | 81.52% | 81.89% | **+0.38** | 81 | 53 |

### Buckets ranked by ΔR@1 (largest gain first)

| Rank | Bucket (in-plane °) | N | BoQ R@1 | DR-VPR R@1 | Δ R@1 |
|---:|---|---:|---:|---:|---:|
| 1 | [ 0°,  5°) | 4662 | 80.76% | 81.60% | **+0.84** |
| 2 | [ 5°, 10°) | 1857 | 83.20% | 83.31% | **+0.11** |
| 3 | [20°, 25°) | 54 | 50.00% | 50.00% | **+0.00** |
| 4 | 25°+ | 6 | 100.00% | 100.00% | **+0.00** |
| 5 | [10°, 15°) | 711 | 85.65% | 84.39% | **-1.27** |
| 6 | [15°, 20°) | 111 | 72.97% | 69.37% | **-3.60** |