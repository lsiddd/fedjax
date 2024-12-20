import jax
import jax.numpy as jnp

def selective_layer_pruning(params, importance_scores, pruning_thresholds):
    """
    A no-op pruning function that returns the original parameters unchanged.
    
    Args:
        params: The model parameters to consider for pruning.
        importance_scores: Dictionary of importance scores for each layer.
        pruning_thresholds: Dictionary of pruning thresholds for each layer.

    Returns:
        The original parameters, without any modifications.
    """
    return params

def calculate_pruning_thresholds(importance_scores, prune_quantile=0.5):
    """
    Calculate pruning thresholds based on importance scores.
    
    Args:
        importance_scores: Dictionary of importance scores for each layer.
        prune_quantile: Quantile for threshold calculation.
        
    Returns:
        A dictionary with thresholds for each layer.
    """
    layer_values = list(importance_scores.values())
    arr = jnp.array(layer_values)
    threshold_val = jnp.quantile(arr, prune_quantile)
    thresholds = {layer: threshold_val for layer in importance_scores}
    return thresholds


# import jax
# import jax.numpy as jnp

# def selective_layer_pruning(params, importance_scores, pruning_thresholds):
#     """Prune model layers selectively based on importance scores."""
#     pruned_params = {}
#     for layer, weights in params.items():
#         threshold = pruning_thresholds.get(layer, 0.0)
#         layer_importance = importance_scores.get(layer, 0.0)
#         if layer_importance < threshold:
#             pruned_params[layer] = jax.tree_util.tree_map(lambda w: 0.0, weights)
#         else:
#             pruned_params[layer] = weights
#     return pruned_params

# def calculate_pruning_thresholds(importance_scores, prune_quantile=0.5):
#     """Calculate pruning thresholds based on importance scores."""
#     layer_values = list(importance_scores.values())
#     arr = jnp.array(layer_values)
#     threshold_val = jnp.quantile(arr, prune_quantile)
#     thresholds = {layer: threshold_val for layer in importance_scores}
#     return thresholds

