"""Gradient histogram utilities for WandB logging."""

import torch
import numpy as np


def log_gradient_histograms(model, wandb_run, step, prefix=""):
    """
    Log gradient histograms to WandB for all parameters in a model.
    
    Args:
        model: (torch.nn.Module) The model whose gradients to log
        wandb_run: WandB run object for logging
        step: (int) Current training step
        prefix: (str) Prefix for the histogram names (e.g., "actor/", "critic/")
    """
    if wandb_run is None:
        return
    
    # Import wandb here to avoid import issues
    import wandb
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Clean parameter name for WandB (replace dots with slashes)
            clean_name = name.replace('.', '/')
            histogram_name = f"{prefix}gradients/{clean_name}"
            
            # Convert gradient to numpy for WandB
            grad_data = param.grad.detach().cpu().numpy()
            
            # Log histogram to WandB
            wandb_run.log({
                histogram_name: wandb.Histogram(grad_data)
            }, step=step)


def collect_gradient_stats(model):
    """
    Collect gradient statistics for a model.
    
    Args:
        model: (torch.nn.Module) The model whose gradients to analyze
        
    Returns:
        dict: Dictionary containing gradient statistics
    """
    stats = {
        'grad_norm': 0.0,
        'grad_mean': 0.0,
        'grad_std': 0.0,
        'grad_max': 0.0,
        'grad_min': 0.0,
        'total_params': 0,
        'params_with_grad': 0
    }
    
    all_grads = []
    total_norm_squared = 0.0
    
    for param in model.parameters():
        if param.grad is not None:
            grad_data = param.grad.detach().cpu().numpy().flatten()
            all_grads.extend(grad_data)
            total_norm_squared += torch.sum(param.grad.detach() ** 2).item()
            stats['params_with_grad'] += 1
        stats['total_params'] += 1
    
    if all_grads:
        all_grads = np.array(all_grads)
        stats['grad_norm'] = np.sqrt(total_norm_squared)
        stats['grad_mean'] = np.mean(all_grads)
        stats['grad_std'] = np.std(all_grads)
        stats['grad_max'] = np.max(all_grads)
        stats['grad_min'] = np.min(all_grads)
    
    return stats


def log_gradient_histograms_with_stats(model, wandb_run, step, prefix="", log_stats=True):
    """
    Log gradient histograms and statistics to WandB for all parameters in a model.
    
    Args:
        model: (torch.nn.Module) The model whose gradients to log
        wandb_run: WandB run object for logging
        step: (int) Current training step
        prefix: (str) Prefix for the histogram names (e.g., "actor/", "critic/")
        log_stats: (bool) Whether to also log gradient statistics
    """
    if wandb_run is None:
        return
    
    # Log histograms
    log_gradient_histograms(model, wandb_run, step, prefix)
    
    # Log statistics if requested
    if log_stats:
        stats = collect_gradient_stats(model)
        for stat_name, stat_value in stats.items():
            wandb_run.log({
                f"{prefix}gradient_stats/{stat_name}": stat_value
            }, step=step) 