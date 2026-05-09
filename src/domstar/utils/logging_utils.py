"""Logging and runtime inspection helpers."""

from __future__ import annotations

import json
import logging
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

import torch


def setup_logging(level: str = "INFO", log_file: str = "") -> logging.Logger:
    """Configure root logging once for CLI scripts."""

    logger = logging.getLogger("domstar")
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logger.setLevel(resolved_level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if not logger.handlers:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        existing_paths = {
            getattr(handler, "baseFilename", None)
            for handler in logger.handlers
            if isinstance(handler, logging.FileHandler)
        }
        if str(log_path.resolve()) not in existing_paths:
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def log_runtime_environment(logger: logging.Logger) -> None:
    """Emit enough runtime detail to debug hardware-specific failures."""

    logger.info("Python %s on %s", platform.python_version(), platform.platform())
    logger.info("Torch %s | CUDA available=%s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info("CUDA device count=%s", device_count)
        for device_index in range(device_count):
            properties = torch.cuda.get_device_properties(device_index)
            total_gb = properties.total_memory / (1024**3)
            logger.info(
                "GPU %s | name=%s | total_vram_gb=%.2f | capability=%s.%s",
                device_index,
                properties.name,
                total_gb,
                properties.major,
                properties.minor,
            )
        logger.info("BF16 supported=%s", torch.cuda.is_bf16_supported())
    else:
        logger.warning("CUDA is not available. Large-model work will be CPU-bound.")

    if cuda_visible_devices := os.environ.get("CUDA_VISIBLE_DEVICES"):
        logger.info("CUDA_VISIBLE_DEVICES=%s", cuda_visible_devices)


def validate_non_empty(name: str, size: int) -> None:
    """Fail fast on empty datasets or empty prepared examples."""

    if size <= 0:
        raise ValueError(f"{name} is empty; nothing to run.")


def prune_checkpoints(output_dir: str | Path, logger: logging.Logger) -> None:
    """Delete intermediate Trainer checkpoint folders after a final save."""

    output_path = Path(output_dir)
    removed = 0
    for checkpoint_dir in output_path.glob("checkpoint-*"):
        if checkpoint_dir.is_dir():
            shutil.rmtree(checkpoint_dir, ignore_errors=False)
            removed += 1
    if removed:
        logger.info("Removed %s intermediate checkpoint folder(s) from %s", removed, output_path)


def _json_safe(value: Any) -> Any:
    """Coerce trainer history values into JSON-safe primitives."""

    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def save_training_history(output_dir: str | Path, log_history: list[dict[str, Any]], logger: logging.Logger) -> Path:
    """Persist Trainer.log_history so runs can be graphed later."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    history_path = output_path / "log_history.json"
    history_payload = _json_safe(log_history)
    history_path.write_text(json.dumps(history_payload, indent=2), encoding="utf-8")
    logger.info("Saved trainer log history to %s", history_path)
    return history_path
