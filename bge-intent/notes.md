# BGE EVA Intent Model: Method and Results

## Run covered

This note describes the revised-validation run saved under `runs/curves/`. It was trained after the validation examples were rewritten to vary their phrasing. The default artifacts in `runs/` were left unchanged.

## Data and split

The benchmark has 3,600 single-intent utterances across 36 labels, with 100 rows per label. The split is stratified by intent:

| Split | Rows | Per intent | Use |
|---|---:|---:|---|
| Training | 2,880 | 80 | Fit model parameters |
| Validation | 360 | 10 | Track learning and select checkpoint |
| Test | 360 | 10 | Final held-out evaluation |

The files used were `../benchmark/data/train.jsonl`, `../benchmark/data/val.jsonl`, and `../benchmark/data/test.jsonl`. The validation set has revised phrasing; the training and test rows were unchanged. The benchmark split audit reported zero gates, including no same-label train-to-held-out pairs at or above its 0.65 character-trigram Jaccard threshold. These are generated benchmark utterances, not a sample of independently collected astronaut speech.

## Method

- **Base model:** `BAAI/bge-small-en-v1.5`.
- **Classifier:** fine-tuned BGE encoder with CLS-token pooling, dropout 0.1, and a linear layer for 36 intent classes. The label is the class with the highest softmax score.
- **Input:** tokenizer truncation at 256 tokens.
- **Objective:** cross-entropy loss.
- **Optimizer:** AdamW, learning rate `2e-5`, weight decay `0.01`.
- **Batch size:** 16.
- **Seed:** 42 for Python, NumPy, and PyTorch.
- **Training schedule:** six epochs requested, patience 2. Checkpoint selection compares validation accuracy first, then validation macro-F1. The best checkpoint was epoch 6.
- **Runtime:** offline, using the locally cached BGE weights; this run used CPU.

The run command, from this directory, was:

```sh
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python train.py \
  ../benchmark/data/train.jsonl \
  --validation-data ../benchmark/data/val.jsonl \
  --test-data ../benchmark/data/test.jsonl \
  --output-dir runs/curves --epochs 6
python plot_history.py runs/curves
```

## Results

At epoch 6, training loss was **0.2217**, validation loss **0.4206**, validation accuracy **89.44%** (322/360), and validation macro-F1 **88.95%**. On the held-out test set, accuracy was **99.44%** (358/360) and macro-F1 was **99.44%**.

| Epoch | Train loss | Validation loss | Validation accuracy | Validation macro-F1 |
|---:|---:|---:|---:|---:|
| 1 | 2.8791 | 2.0585 | 78.61% | 76.67% |
| 2 | 1.4717 | 1.1575 | 83.89% | 82.81% |
| 3 | 0.7758 | 0.7758 | 85.28% | 84.41% |
| 4 | 0.4510 | 0.5735 | 88.33% | 87.89% |
| 5 | 0.3044 | 0.4898 | 88.89% | 88.41% |
| 6 | 0.2217 | 0.4206 | 89.44% | 88.95% |

Validation performance is lower than the test result because the revised validation phrases are more varied; the test split was not revised. Keep the two figures distinct and use the validation result to assess performance on the revised phrasing. The model still missed two `Add_waypoint` test rows (80% recall for that class); `Set_navigation_target` had 83% test precision. Per-class metrics are in `runs/curves/metrics.json`.

## Curves and artifacts

- [Validation learning curves](runs/curves/learning_curves.svg)
- [Training and validation loss curves](runs/curves/loss_curves.svg)
- [Per-epoch metrics and test report](runs/curves/metrics.json)
- Model weights: `runs/curves/model.pt`
- Label mapping: `runs/curves/label2id.json`
