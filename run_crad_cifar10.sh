#!/usr/bin/env bash
# Run CRAD on CIFAR-10 (Paper Benchmark)
python experiments/main.py \
  --algorithm TypicalKD \
  --idea 4.3_pre_v0 \
  --adaptive_pre_filter \
  --val_test_protocol option_a \
  --k_zero_strategy drop \
  --fix_eq6 \
  --dataset cifar10 \
  --num_clients 10 \
  --num_rounds 300 \
  --alpha_kd 0.7 \
  --partition_alpha 0.3 \
  --T 4.0 \
  --cs full \
  --lr 0.1 \
  --seed 1024 \
  --use_global_test \
  --fast_eval \
  --name "CRAD_cifar10_paper" "$@"
