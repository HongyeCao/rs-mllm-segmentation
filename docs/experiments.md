## E01 LoveDA V1 Baseline — Epoch 1

Training code commit:
910994cbd949741040580135a0df70b50936128b

Train patches:
22695

Effective supervised microbatches:
22650

Validation patches:
15020

Seed:
42

Learning rate:
1e-4

Gradient accumulation:
8

Epochs:
1

Checkpoint:
epoch_01_step_0022650

OA:
66.1174

mIoU:
45.5297

Per-class IoU:
Background: 49.2680
Building: 56.2408
Road: 47.1205
Water: 56.6987
Barren: 17.5158
Forest: 37.1868
Agriculture: 54.6777

Status:
Valid baseline; epoch-2 continuation recommended.


## E01 LoveDA V1 Baseline — Epoch 2

Checkpoint:
epoch_02_step_0045300

OA:
67.5192

mIoU:
47.0197

Per-class IoU:
Background: 51.8478
Building: 55.9465
Road: 49.7696
Water: 58.4282
Barren: 22.4925
Forest: 36.7252
Agriculture: 53.9281

Change from Epoch 1:
OA: +1.4018
mIoU: +1.4900

Current best:
Epoch 2

Status:
Improved; one additional epoch recommended for convergence check.


## D01 — LoveDA 10% Label-Efficiency Baseline

Branch:
exp/d01-loveda-data-efficiency

Training split:
LoveDA 10%, source-image-grouped sampling, seed=42

Effective training samples per epoch:
2273

Training epochs:
10

Best epoch:
9

Best checkpoint:
experiments/D01_LoveDA_10pct_seed42/epoch_09_step_0020457

Validation samples:
15020

Best validation results:
OA: 65.3105
mIoU: 44.5325

Per-class IoU:
Background: 49.8338
Building: 52.2964
Road: 44.7316
Water: 57.6525
Barren: 24.7824
Forest: 31.6500
Agriculture: 50.7807

Full-data V1 baseline:
OA: 67.5192
mIoU: 47.0197

Performance retention:
94.71%

Best-checkpoint optimization microsteps:
20457

Full-data baseline optimization microsteps:
45300

Relative optimization budget:
45.16%

Conclusion:
Using approximately 10% of the labeled LoveDA training data,
the Qwen2-VL V1 baseline retains approximately 94.7% of the
full-data segmentation performance. The result demonstrates
strong label efficiency, although comparison with conventional
segmentation baselines is still required before attributing the
advantage specifically to the multimodal large model.
Agriculture: 50.7807

Full-data V1 baseline:
OA: 67.5192
mIoU: 47.0197

Performance retention:
94.71%

Best-checkpoint optimization microsteps:
20457

Full-data baseline optimization microsteps:
45300

Relative optimization budget:
45.16%

Conclusion:
Using approximately 10% of the labeled LoveDA training data,
the Qwen2-VL V1 baseline retains approximately 94.7% of the
full-data segmentation performance. The result demonstrates
strong label efficiency, although comparison with conventional
segmentation baselines is still required before attributing the
advantage specifically to the multimodal large model.
