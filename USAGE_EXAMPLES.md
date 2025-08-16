# HGSF-DTA Usage Examples

This document provides detailed examples for running HGSF-DTA experiments.

## Quick Start

1. **Setup Environment**

   ```bash
   # On Linux/Mac
   bash setup.sh

   # On Windows
   setup.bat
   ```

2. **Activate Environment**
   ```bash
   conda activate hgsf-dta
   cd source
   ```

## Experiment Examples

### Example 1: Basic Training on Davis Dataset

```bash
# Standard scenario with default parameters
python train1.py --dataset davis --cuda_id 0 --num_epochs 1000 --batch_size 256 --lr 0.0001

# Expected output:
# Dataset: davis
# Cuda name: cuda:0
# Epochs: 1000
# Learning rate: 0.0001
# Model name: BGNN
# Train and test
```

### Example 2: Cross-Validation on KIBA Dataset

```bash
# Run 5-fold cross-validation
for fold in 0 1 2 3 4; do
    echo "Training fold $fold..."
    python train1.py --dataset kiba --fold $fold --num_epochs 1000 --batch_size 256 --lr 0.0001
done
```

### Example 3: Cold Drug Scenario

```bash
# S2 scenario - testing on unseen drugs
python train2.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 256 --lr 0.0001 --target_aff_k 150

# For KIBA dataset, use different target_aff_k
python train2.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 256 --lr 0.0001 --target_aff_k 90
```

### Example 4: Cold Protein Scenario

```bash
# S3 scenario - testing on unseen targets
python train3.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 256 --lr 0.0001 --target_aff_k 150
```

### Example 5: Cold Pair Scenario

```bash
# S4 scenario - testing on unseen drug-target pairs
python train4.py --dataset davis --cuda_id 0 --num_epochs 200 --batch_size 256 --lr 0.0001 --target_aff_k 150

# For KIBA dataset
python train4.py --dataset kiba --cuda_id 0 --num_epochs 200 --batch_size 256 --lr 0.0001 --target_aff_k 90
```

### Example 6: Hyperparameter Tuning

```bash
# Different learning rates
python train1.py --dataset davis --lr 0.001 --num_epochs 500
python train1.py --dataset davis --lr 0.0005 --num_epochs 750
python train1.py --dataset davis --lr 0.0001 --num_epochs 1000

# Different batch sizes
python train1.py --dataset davis --batch_size 128 --num_epochs 1000
python train1.py --dataset davis --batch_size 512 --num_epochs 1000

# Different similarity parameters
python train1.py --dataset davis --drug_sim_k 3 --target_sim_k 10 --drug_aff_k 50 --target_aff_k 200
```

### Example 7: CPU-only Training

```bash
# For systems without CUDA-enabled GPU
python train1.py --dataset davis --cuda_id -1 --batch_size 128 --num_epochs 500
```

## Expected Training Logs

```
Dataset: davis
Cuda name: cuda:0
Epochs: 1000
Learning rate: 0.0001
Model name: BGNN
Train and test
create dataset ...
len(train_fold_origin)===== 5
create train_loader and test_loader ...
create drug_graphs_dict and target_graphs_dict ...
create drug_graphs_DataLoader and target_graphs_DataLoader ...
create affinity_graph ...
device: cuda:0
epoch: 0, lr: 0.0001, loss: 1.2345, MSE: 0.678, CI: 0.567, r2: 0.123, Pearson: 0.456, AUPR: 0.789
epoch: 1, lr: 0.0001, loss: 1.1234, MSE: 0.645, CI: 0.589, r2: 0.145, Pearson: 0.478, AUPR: 0.812
...
```

## Model Output Files

After training, the following files will be created:

```
models/
├── architecture/
│   └── davis/
│       └── S1/
│           ├── cross_validation/
│           └── test/
└── predictor/
    └── davis/
        └── S1/
            ├── cross_validation/
            └── test/
```

## Performance Monitoring

To monitor GPU usage during training:

```bash
nvidia-smi -l 1
```

To monitor system resources:

```bash
htop
```

## Troubleshooting Common Issues

### Issue 1: CUDA Out of Memory

```bash
# Reduce batch size
python train1.py --dataset davis --batch_size 128

# Or use CPU
python train1.py --dataset davis --cuda_id -1
```

### Issue 2: Slow Training

```bash
# Increase batch size (if memory allows)
python train1.py --dataset davis --batch_size 512

# Reduce number of epochs for testing
python train1.py --dataset davis --num_epochs 100
```

### Issue 3: Poor Convergence

```bash
# Try different learning rate
python train1.py --dataset davis --lr 0.001

# Adjust similarity parameters
python train1.py --dataset davis --drug_sim_k 1 --target_sim_k 5
```

## Reproducing Paper Results

To reproduce the exact results from the paper:

1. **Davis S1 scenario:**

   ```bash
   python train1.py --dataset davis --fold -100 --num_epochs 1000 --batch_size 256 --lr 0.0001 --drug_sim_k 2 --target_sim_k 7 --drug_aff_k 40 --target_aff_k 150
   ```

2. **KIBA S2 scenario:**

   ```bash
   python train2.py --dataset kiba --fold -100 --num_epochs 200 --batch_size 256 --lr 0.0001 --drug_sim_k 2 --target_sim_k 7 --drug_aff_k 40 --target_aff_k 90
   ```

3. **Davis 5-fold CV:**
   ```bash
   for fold in 0 1 2 3 4; do
       python train1.py --dataset davis --fold $fold --num_epochs 1000 --batch_size 256 --lr 0.0001
   done
   ```
