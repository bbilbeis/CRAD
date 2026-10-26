#!/usr/bin/env python3
"""Run CRAD on CIFAR-10 (Paper Benchmark) - zero arguments needed."""
import os
import sys
import subprocess

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MAIN_SCRIPT = os.path.join(ROOT_DIR, "experiments", "main.py")

cmd = [
    sys.executable, MAIN_SCRIPT,
    "--algorithm", "TypicalKD",
    "--idea", "4.3_pre_v0",
    "--adaptive_pre_filter",
    "--val_test_protocol", "option_a",
    "--k_zero_strategy", "drop",
    "--fix_eq6",
    "--dataset", "cifar10",
    "--num_clients", "10",
    "--num_rounds", "300",
    "--alpha_kd", "0.7",
    "--partition_alpha", "0.3",
    "--T", "4.0",
    "--cs", "full",
    "--lr", "0.1",
    "--seed", "1024",
    "--use_global_test",
    "--fast_eval",
    "--name", "CRAD_cifar10_paper"
] + sys.argv[1:]

sys.exit(subprocess.call(cmd, cwd=ROOT_DIR))
