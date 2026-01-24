import math
import numpy as np
import pandas as pd

def select_anchors(predictions, label_map_inv, config, logger, **kwargs):
    """
    Unified entry point for anchor selection.
    Dispatches to different strategy functions based on 'method' in config.
    
    Args:
        predictions (dict): LLM prediction dictionary {node_id: {llm_predict: str, llm_confident: float}}
        label_map_inv (dict): Label name to index mapping {'CS.AI': 0, ...}
        config (dict): data_filter configuration section
        logger: Logger object
        **kwargs: Extra arguments like split_idx, gt_y (used for supervised strategy)
    
    Returns:
        pseudo_indices (list): Selected node IDs
        pseudo_labels (list): Corresponding pseudo label indices
    """
    method = config.get('method', 'confidence_threshold')
    logger.info(f"Applying Anchor Strategy: {method} with params {config}")

    # 1. Preprocessing: Convert Dict to DataFrame for vectorized filtering and sorting
    #    This is more efficient than the previous loop for complex strategies
    data = []
    valid_predictions = 0
    
    for nid_str, res in predictions.items():
        nid = int(nid_str)
        conf = res.get('llm_confident', -999)
        # Handle potential LogProb or Probability
        prob = conf if conf > 0 else math.exp(conf)
        if prob > 1.0: prob = 1.0
        
        # Normalize predicted string
        pred_str = res.get('llm_predict', '').upper().replace('ARXIV ', '').replace(' ', '.')
        
        if pred_str in label_map_inv:
            cat_idx = label_map_inv[pred_str]
            data.append({
                'nid': nid,
                'prob': prob,
                'pred_idx': cat_idx
            })
            valid_predictions += 1

    df = pd.DataFrame(data)
    logger.info(f"Total valid candidates for anchoring: {len(df)}")

    # 2. Strategy Dispatch
    if method == 'confidence_threshold':
        selected_df = _strategy_threshold(df, config)
    elif method == 'top_k':
        selected_df = _strategy_top_k(df, config)
    elif method == 'bounded_proportional':
        selected_df = _strategy_bounded_proportional(df, config)
    elif method == 'all':
        selected_df = _strategy_all(df, config)
    elif method == 'supervised':
        selected_df = _strategy_supervised(config, logger, **kwargs)
    else:
        logger.error(f"Unknown strategy method: {method}")
        return [], []

    # 3. Format Output
        
    pseudo_indices = selected_df['nid'].tolist()
    pseudo_labels = selected_df['pred_idx'].tolist()
    
    # Simple statistical report
    counts = selected_df['pred_idx'].value_counts()
    min_c, max_c = counts.min(), counts.max()
    logger.info(f"Selected {len(pseudo_indices)} anchors. Class distribution: Min={min_c}, Max={max_c}")
    
    return pseudo_indices, pseudo_labels

# --- Specific Strategy Implementations ---

def _strategy_all(df, config):
    """
    Strategy: Use all predictions (Baseline 2 - Full Trust)
    """
    return df

def _strategy_threshold(df, config):
    """
    Strategy 1: Simple Threshold Filtering
    params: threshold (float)
    """
    thresh = config.get('threshold', 0.8)
    return df[df['prob'] >= thresh]

def _strategy_top_k(df, config):
    """
    Strategy 2: Select Top-K per Class
    params: k (int)
    """
    k = config.get('k', 300)
    # Sort by confidence descending
    df_sorted = df.sort_values('prob', ascending=False)
    # Group by class and take Top K
    return df_sorted.groupby('pred_idx').head(k)

def _strategy_bounded_proportional(df, config):
    """
    Strategy 4: Bounded Proportional Selection (Recommended)
    params: 
        threshold (float): Basic entry threshold (default 0.99)
        top_pct (float): Select top X% of high-confidence samples (default 0.2 i.e. 20%)
        min_n (int): Minimum count per class (default 50)
        max_n (int): Maximum count per class (default 1000)
    """
    threshold = config.get('threshold', 0.99)
    top_pct = config.get('top_pct', 0.2)
    min_n = config.get('min_n', 50)
    max_n = config.get('max_n', 1000)

    # 1. Basic filtering
    mask = df['prob'] >= threshold
    df_c = df[mask].sort_values('prob', ascending=False)
    
    final_indices = []
    
    # 2. Group and dynamically calculate N
    for cls_idx, group in df_c.groupby('pred_idx'):
        total_candidates = len(group)
        
        # Calculate target count: X% of candidates
        n_target = int(total_candidates * top_pct)
        # Apply bound constraints [min_n, max_n]
        n_final = max(min_n, min(n_target, max_n))
        
        # If candidates are fewer than min_n, take all
        if total_candidates < n_final:
            n_final = total_candidates
            
        final_indices.extend(group.head(n_final).index.tolist())
        
    return df.loc[final_indices]
    
def _strategy_supervised(config, logger, split_idx=None, gt_y=None, **kwargs):
    """
    Strategy: Baseline 3 - Fully Supervised (Ground Truth)
    Directly returns the true labels of the training set
    """
    if split_idx is None or gt_y is None:
        logger.error("Supervised strategy requires 'split_idx' and 'gt_y' to be passed!")
        return pd.DataFrame()
        
    logger.info("[BASELINE] Using Ground Truth labels for Training Set")
    
    train_idx = split_idx['train']
    if isinstance(train_idx, np.ndarray):
        indices = train_idx.tolist()
    else:
        indices = train_idx.tolist() # tensor to list
        
    # extract labels
    # gt_y is tensor [N, 1] or [N]
    labels_tensor = gt_y[indices]
    if labels_tensor.dim() > 1:
        labels_tensor = labels_tensor.squeeze()
        
    labels = labels_tensor.tolist()
    
    logger.info(f"Selected {len(indices)} Ground Truth anchors.")
    
    # Return as DataFrame to match other strategies
    return pd.DataFrame({
        'nid': indices,
        'pred_idx': labels,
        'prob': 1.0
    })
