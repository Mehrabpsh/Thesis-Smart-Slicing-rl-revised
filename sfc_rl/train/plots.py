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
    metrics = ["acceptance_ratio", "qoe"]#, "response_time"]
    
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
    losses: List[float],
    output_dir: Path,
    filename: str = "training_curve.png",
    window: int = 100,
    **kwargs
) -> None:
    """Plot training curves for rewards and losses.
    
    Args:
        rewards: List of episode rewards
        losses: List of training losses
        output_dir: Output directory
        filename: Output filename
        window: Moving average window
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = kwargs.get('episode', None)
    notebook = kwargs.get('run_in_notebook', False)

    if len(rewards) == 0 and len(losses) == 0:
        return
    
    fig, axes = plt.subplots(1, 1, figsize=(12, 5))
    
    # # Plot 1: Rewards
    # if len(rewards) > 0:
    #     axes[0].plot(list(range(1, len(rewards) + 1)), rewards, alpha=0.3, label="Raw", color="blue")
        
    #     # Moving average for rewards
    #     if len(rewards) >= window:
    #         moving_avg = pd.Series(rewards).rolling(window=window).mean()
    #         axes[0].plot(moving_avg, label=f"Moving Avg ({window})", color="red", linewidth=2)
        
    #     axes[0].set_xlabel("Episode")
    #     axes[0].set_ylabel("qoe")
    #     axes[0].set_title("qoe Progress")
    #     axes[0].legend()
    #     axes[0].grid(True, alpha=0.3)
    # else:
    #     axes[0].text(0.5, 0.5, "No reward data", ha='center', va='center', transform=axes[0].transAxes)
    #     axes[0].set_title("qoe Progress")
    
    # Plot 2: Losses
    if len(losses) > 0:
        # For losses, we might have more data points than episodes
        #x_axis = range(len(losses)//100)
        
        x_axis = np.linspace(0,episode,len(losses))

        axes[1].plot(x_axis, losses, alpha=0.3, label="Raw", color="green")
        
        # Moving average for losses (with smaller window since there might be more points)
        loss_window = min(window, len(losses) // 10) if len(losses) > 10 else len(losses)
        if len(losses) >= loss_window:
            moving_avg_loss = pd.Series(losses).rolling(window=loss_window).mean()
            axes[1].plot(x_axis, moving_avg_loss, label=f"Moving Avg ({loss_window})", color="red", linewidth=2)
        
        axes[1].set_xlabel("Episode")
        axes[1].set_ylabel("Loss")
        axes[1].set_title("Loss Progress")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    else:
        axes[1].text(0.5, 0.5, "No loss data", ha='center', va='center', transform=axes[1].transAxes)
        axes[1].set_title("Loss Progress")
    
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
    
    if notebook:
        plt.show() 
    
    plt.close()