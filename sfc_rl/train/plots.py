"""Plotting utilities for metrics visualization."""

from typing import Dict, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_metrics(
    metrics_dict: Dict[str, Dict[str, Any]],
    output_dir: Path,
    filename: str = "metrics_comparison.png",
) -> None:
    """Plot metrics comparison across policies.
    
    Args:
        metrics_dict: Dictionary mapping policy names to metrics
        output_dir: Output directory
        filename: Output filename
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    policies = list(metrics_dict.keys())
    metrics = ["acceptance_ratio", "qoe", "response_time"]
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
    
    for idx, metric in enumerate(metrics):
        values = [metrics_dict[policy].get(metric, 0.0) for policy in policies]
        axes[idx].bar(policies, values)
        axes[idx].set_title(metric.replace("_", " ").title())
        axes[idx].set_ylabel(metric.replace("_", " ").title())
        axes[idx].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()


def plot_training_curves(
    rewards: List[float],
    output_dir: Path,
    filename: str = "training_curve.png",
    window: int = 100,
) -> None:
    """Plot training curve.
    
    Args:
        rewards: List of episode rewards
        output_dir: Output directory
        filename: Output filename
        window: Moving average window
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if len(rewards) == 0:
        return
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Raw rewards
    ax.plot(rewards, alpha=0.3, label="Raw", color="blue")
    
    # Moving average
    if len(rewards) >= window:
        moving_avg = pd.Series(rewards).rolling(window=window).mean()
        ax.plot(moving_avg, label=f"Moving Avg ({window})", color="red", linewidth=2)
    
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reward")
    ax.set_title("Training Curve")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    plt.close()

