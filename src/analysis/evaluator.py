import os
import json
import torch
import math
import numpy as np
from tqdm import tqdm

def evaluate_results(data, split_idx, loader, output_dir, logger, config=None):
    """
    Load predictions from disk, compare with Ground Truth, and compute metrics.
    
    Args:
        data: PyG data object (containing Ground Truth y).
        split_idx: Dictionary of train/valid/test indices.
        loader: DataLoader instance (for label mapping).
        output_dir: Experiment output directory.
        logger: Logger instance.
        config: Full configuration dict (optional).
    """
    logger.info("Starting Evaluation & Analysis...")
    
    # Ensemble Settings
    ensemble_active = False
    threshold = 1.0
    if config and 'ensemble' in config:
        ensemble_active = True
        threshold = config['ensemble'].get('threshold', 0.99)
        logger.info(f"[Ensemble] Active. Threshold={threshold}. Logic: IF LLM_Conf > Thresh THEN LLM ELSE GNN")
    
    # paths
    llm_path = os.path.join(output_dir, "llm_predictions.json")
    gnn_path = os.path.join(output_dir, "gnn_predictions.json")
    
    # 1. Load LLM Predictions
    llm_preds = {}
    if os.path.exists(llm_path):
        with open(llm_path, 'r') as f:
            raw_llm = json.load(f)
            # Normalize keys to int
            for k, v in raw_llm.items():
                llm_preds[int(k)] = v
    else:
        logger.warning(f"LLM predictions not found at {llm_path}")

    # 2. Load GNN Predictions
    gnn_preds = {}
    if os.path.exists(gnn_path):
        with open(gnn_path, 'r') as f:
            raw_gnn = json.load(f)
            for k, v in raw_gnn.items():
                gnn_preds[int(k)] = v
    else:
        logger.warning(f"GNN predictions not found at {gnn_path}")

    # Helper for label mapping
    def prepare_label_mapping(loader):
        label_map_inv = {}
        if loader.label_map:
            for idx, cat_str in loader.label_map.items():
                clean_cat = cat_str.upper().replace('ARXIV ', '').replace(' ', '.')
                label_map_inv[clean_cat] = idx
        return label_map_inv
    
    inv_map = prepare_label_mapping(loader)
    
    # Helper to get clean category string
    def get_cat_str(cat_idx):
        if not loader.label_map:
            return str(cat_idx)
        s = loader.label_map.get(cat_idx, str(cat_idx))
        return str(s).replace('arxiv ', '')

    # Prepare stats containers
    gt = data.y.squeeze().cpu().numpy()
    results_list = []
    
    # Map node to subset
    node_to_subset = {}
    for split_name, indices in split_idx.items():
        if torch.is_tensor(indices):
            idx_np = indices.cpu().numpy()
        else:
            idx_np = np.array(indices)
        for idx in idx_np:
            node_to_subset[int(idx)] = split_name

    # Tracking metrics
    correct_count = 0
    llm_correct_count = 0
    gnn_correct_count = 0
    total_count = 0
    
    split_stats = {
        k: {'correct': 0, 'llm_correct': 0, 'gnn_correct': 0, 'total': 0} for k in split_idx.keys()
    }
    split_stats['unknown'] = {'correct': 0, 'llm_correct': 0, 'gnn_correct': 0, 'total': 0}

    # Iterate all nodes
    for idx_val in tqdm(range(data.num_nodes), desc="Evaluating"):
        idx = int(idx_val)
        subset = node_to_subset.get(idx, 'unknown')
        ground_truth_idx = int(gt[idx])
        ground_truth_cat = get_cat_str(ground_truth_idx)
        
        # Resolve LLM info
        llm_res = llm_preds.get(idx, {})
        llm_cat_str = llm_res.get('llm_predict', llm_res.get('category'))
        llm_raw_conf = llm_res.get('llm_confident', llm_res.get('confidence', -999))
        
        # Normalize confidence (Same logic as selector.py)
        llm_prob = llm_raw_conf
        if llm_prob <= 0: # assume logprob
             llm_prob = math.exp(llm_raw_conf)
        if llm_prob > 1.0: llm_prob = 1.0
        
        # Resolve GNN info
        gnn_res = gnn_preds.get(idx, {})
        gnn_pred_idx = gnn_res.get('gnn_predict')
        
        has_llm = (llm_cat_str is not None)
        has_gnn = (gnn_res and gnn_pred_idx is not None)

        # Standalone Metric Check
        if has_llm:
            clean_llm = str(llm_cat_str).upper()
            if clean_llm in inv_map:
                if inv_map[clean_llm] == ground_truth_idx:
                    llm_correct_count += 1
                    split_stats[subset]['llm_correct'] += 1
        
        if has_gnn:
            if int(gnn_pred_idx) == ground_truth_idx:
                gnn_correct_count += 1
                split_stats[subset]['gnn_correct'] += 1

        # Determine Final Prediction strategy
        final_pred_idx = -1
        final_pred_cat = "N/A"
        is_correct = False
        
        use_source = 'none'
        
        if has_llm and has_gnn:
            # Both exist: Check Ensemble
            if ensemble_active and llm_prob > threshold:
                use_source = 'llm'
            else:
                use_source = 'gnn'
        elif has_gnn:
            use_source = 'gnn'
        elif has_llm:
            use_source = 'llm'
            
        # Apply selection
        if use_source == 'gnn':
            # Case A: Use GNN result
            final_pred_idx = int(gnn_pred_idx)
            is_correct = (final_pred_idx == ground_truth_idx)
            final_pred_cat = get_cat_str(final_pred_idx)
            
        elif use_source == 'llm':
            # Case B: Use LLM result
            # Normalize prompt output
            clean_llm = str(llm_cat_str).upper()
            if clean_llm in inv_map:
                final_pred_idx = inv_map[clean_llm]
                is_correct = (final_pred_idx == ground_truth_idx)
                final_pred_cat = get_cat_str(final_pred_idx)
            else:
                # LLM predicted something invalid or unparsable
                is_correct = False
                final_pred_cat = f"Invalid({llm_cat_str})"
        
        # Update Stats
        total_count += 1
        if is_correct:
            correct_count += 1
        
        split_stats[subset]['total'] += 1
        if is_correct:
            split_stats[subset]['correct'] += 1
            
        # Record Detail
        results_list.append({
            "node_idx": idx,
            "subset": subset,
            "llm_predict": llm_cat_str,
            "llm_confident": llm_raw_conf,
            "gnn_predict": get_cat_str(gnn_pred_idx) if gnn_pred_idx is not None else None,
            "final_predict": final_pred_cat,
            "ground_truth": ground_truth_cat,
            "is_correct": is_correct
        })

    # Compile Metrics
    metrics = {
        "overall": {
            "accuracy": float(correct_count / total_count * 100) if total_count > 0 else 0.0,
            "correct": correct_count,
            "llm_accuracy": float(llm_correct_count / total_count * 100) if total_count > 0 else 0.0,
            "gnn_accuracy": float(gnn_correct_count / total_count * 100) if total_count > 0 else 0.0,
            "total": total_count
        }
    }
    
    for split_name, stats in split_stats.items():
        if stats['total'] > 0:
            metrics[split_name] = {
                "accuracy": float(stats['correct'] / stats['total'] * 100),
                "correct": stats['correct'],
                "llm_accuracy": float(stats['llm_correct'] / stats['total'] * 100),
                "gnn_accuracy": float(stats['gnn_correct'] / stats['total'] * 100),
                "total": stats['total']
            }
            logger.info(f"[{split_name.upper()}] Acc: {metrics[split_name]['accuracy']:.4f} | LLM: {metrics[split_name]['llm_accuracy']:.4f} | GNN: {metrics[split_name]['gnn_accuracy']:.4f}")

    logger.info(f"[OVERALL] Acc: {metrics['overall']['accuracy']:.4f} | LLM: {metrics['overall']['llm_accuracy']:.4f} | GNN: {metrics['overall']['gnn_accuracy']:.4f}")

    # Save Results
    # Split into lightweight results.json (metrics) and heavyweight predictions.json (details)
    results_output = metrics
    
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results_output, f, indent=2)

    predictions_path = os.path.join(output_dir, "predictions.json")
    with open(predictions_path, 'w') as f:
        json.dump(results_list, f, indent=2)
        
    logger.info(f"Metrics saved to {results_path}")
    logger.info(f"Predictions saved to {predictions_path}")
    return metrics
