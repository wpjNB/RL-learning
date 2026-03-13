"""
训练日志工具

提供 CSV 记录和控制台输出，可选 TensorBoard 支持。
"""

import csv
import os
import time
from typing import Dict, Optional


class Logger:
    """训练日志记录器，将指标写入 CSV 文件并打印到控制台。"""

    def __init__(self, log_dir: str, filename: str = "train_log.csv") -> None:
        os.makedirs(log_dir, exist_ok=True)
        self.log_path = os.path.join(log_dir, filename)
        self._writer: Optional[csv.DictWriter] = None
        self._file = None
        self._start_time = time.time()
        self._fieldnames: Optional[list] = None

    def log(self, metrics: Dict[str, float], step: int, verbose: bool = True) -> None:
        """记录一组指标。"""
        if self._writer is None:
            self._fieldnames = ["step", "time_elapsed"] + list(metrics.keys())
            self._file = open(self.log_path, "w", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(self._file, fieldnames=self._fieldnames)
            self._writer.writeheader()

        elapsed = time.time() - self._start_time
        row = {"step": step, "time_elapsed": f"{elapsed:.1f}"}
        row.update({k: f"{v:.4f}" for k, v in metrics.items()})
        self._writer.writerow(row)
        self._file.flush()

        if verbose:
            parts = [f"Step {step:6d}", f"Time {elapsed:6.0f}s"]
            parts += [f"{k}: {v:.4f}" for k, v in metrics.items()]
            print(" | ".join(parts))

    def close(self) -> None:
        if self._file is not None:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
