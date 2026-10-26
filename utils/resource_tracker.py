"""
ResourceTracker Utility for CRAD Federated Learning Codebase.

Tracks resource consumption and cost profiling per communication round:
1. Network Payload (Bytes/round): Serialized model parameters and class statistics transmitted across clients.
2. Wall-Clock Time (Seconds/round): Elapsed time per round using time.perf_counter().
3. Peak Memory Usage (MB/round): Peak GPU VRAM (torch.cuda.max_memory_allocated) and CPU RAM (psutil with fallback).

Provides automatic JSON export to results/cost_metrics_<method>_<dataset>.json and a formatted summary table.
"""

import os
import time
import json
import torch
import resource

class ResourceTracker:
    def __init__(self, method: str = "TypicalKD", dataset: str = "cifar10", enabled: bool = True, output_dir: str = "results/cost_tracking", timestamp: str = None):
        self.method = method
        self.dataset = dataset
        self.enabled = enabled
        self.output_dir = output_dir
        self.timestamp = timestamp or time.strftime("%Y%m%d-%H%M%S")
        self.round_metrics = []
        self.current_round = None
        self.current_round_start = None
        self.current_round_payload_bytes = 0.0

    def start_round(self, round_idx: int):
        if not self.enabled:
            return
        self.current_round = round_idx
        self.current_round_payload_bytes = 0.0
        if torch.cuda.is_available():
            try:
                torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
        self.current_round_start = time.perf_counter()

    def add_payload_bytes(self, bytes_count: float):
        if not self.enabled:
            return
        self.current_round_payload_bytes += float(bytes_count)

    def end_round(self) -> dict:
        if not self.enabled:
            return {}
        
        elapsed = time.perf_counter() - self.current_round_start if self.current_round_start is not None else 0.0
        
        # GPU VRAM Peak (MB)
        gpu_peak_mb = 0.0
        if torch.cuda.is_available():
            try:
                gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            except Exception:
                gpu_peak_mb = 0.0
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            try:
                if hasattr(torch, 'mps') and hasattr(torch.mps, 'driver_allocated_memory'):
                    gpu_peak_mb = torch.mps.driver_allocated_memory() / (1024.0 * 1024.0)
                elif hasattr(torch, 'mps') and hasattr(torch.mps, 'current_allocated_memory'):
                    gpu_peak_mb = torch.mps.current_allocated_memory() / (1024.0 * 1024.0)
            except Exception:
                gpu_peak_mb = 0.0

        # CPU RAM Peak (MB) with robust psutil fallback
        cpu_peak_mb = 0.0
        try:
            import psutil
            process = psutil.Process(os.getpid())
            cpu_peak_mb = process.memory_info().rss / (1024.0 * 1024.0)
        except ImportError:
            try:
                # ru_maxrss is in KB on Linux, bytes on macOS
                maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                import sys
                if sys.platform == 'darwin':
                    cpu_peak_mb = maxrss / (1024.0 * 1024.0)
                else:
                    cpu_peak_mb = maxrss / 1024.0
            except Exception:
                cpu_peak_mb = 0.0
        except Exception:
            cpu_peak_mb = 0.0

        metrics = {
            "round": self.current_round,
            "payload_bytes": self.current_round_payload_bytes,
            "payload_mb": self.current_round_payload_bytes / (1024.0 * 1024.0),
            "wall_clock_time_sec": elapsed,
            "peak_gpu_mem_mb": gpu_peak_mb,
            "peak_cpu_mem_mb": cpu_peak_mb,
        }
        self.round_metrics.append(metrics)
        return metrics

    def get_summary(self) -> dict:
        if not self.round_metrics:
            return {
                "method": self.method,
                "dataset": self.dataset,
                "total_rounds": 0,
                "avg_payload_bytes_per_round": 0.0,
                "avg_payload_mb_per_round": 0.0,
                "avg_wall_clock_sec_per_round": 0.0,
                "total_wall_clock_sec": 0.0,
                "peak_gpu_mem_mb": 0.0,
                "peak_cpu_mem_mb": 0.0,
            }

        total_rounds = len(self.round_metrics)
        total_payload = sum(m["payload_bytes"] for m in self.round_metrics)
        total_time = sum(m["wall_clock_time_sec"] for m in self.round_metrics)
        max_gpu = max(m["peak_gpu_mem_mb"] for m in self.round_metrics)
        max_cpu = max(m["peak_cpu_mem_mb"] for m in self.round_metrics)

        return {
            "method": self.method,
            "dataset": self.dataset,
            "total_rounds": total_rounds,
            "avg_payload_bytes_per_round": total_payload / total_rounds,
            "avg_payload_mb_per_round": (total_payload / total_rounds) / (1024.0 * 1024.0),
            "total_payload_mb": total_payload / (1024.0 * 1024.0),
            "avg_wall_clock_sec_per_round": total_time / total_rounds,
            "total_wall_clock_sec": total_time,
            "peak_gpu_mem_mb": max_gpu,
            "peak_cpu_mem_mb": max_cpu,
        }

    def save_json(self, filepath: str = None) -> str:
        if not self.enabled:
            return ""
        os.makedirs(self.output_dir, exist_ok=True)
        if filepath is None:
            filename = f"cost_metrics_{self.method}_{self.dataset}_{self.timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)

        summary = self.get_summary()
        data = {
            "summary": summary,
            "rounds": self.round_metrics,
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        print(f"📊 [ResourceTracker] Cost metrics saved to {filepath}")
        
        # Also write a latest symlink/file for convenience
        try:
            latest_path = os.path.join(self.output_dir, f"cost_metrics_{self.method}_{self.dataset}_latest.json")
            with open(latest_path, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

        return filepath

    def print_summary_table(self):
        if not self.enabled or not self.round_metrics:
            return
        summary = self.get_summary()
        title = f"RESOURCE USAGE & COST PROFILING ({summary['method']} on {summary['dataset']})"
        width = 85
        print("\n" + "=" * width)
        print(f"{title:^{width}}")
        print("=" * width)
        print(f" Total Rounds Tracked           : {summary['total_rounds']}")
        print(f" Total Wall-Clock Time          : {summary['total_wall_clock_sec']:.2f} s ({summary['total_wall_clock_sec']/60.0:.2f} min)")
        print(f" Average Wall-Clock Time/Round : {summary['avg_wall_clock_sec_per_round']:.3f} s")
        print(f" Average Network Payload/Round  : {summary['avg_payload_mb_per_round']:.3f} MB ({summary['avg_payload_bytes_per_round']:,.0f} Bytes)")
        print(f" Total Transmitted Payload      : {summary['total_payload_mb']:.2f} MB")
        print(f" Peak GPU Memory Usage          : {summary['peak_gpu_mem_mb']:.2f} MB")
        print(f" Peak CPU RAM Usage             : {summary['peak_cpu_mem_mb']:.2f} MB")
        print("=" * width + "\n")
