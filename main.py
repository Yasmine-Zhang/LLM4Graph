import argparse
import os
import json
import torch
import math
import numpy as np
from tqdm import tqdm
from src.data_loader.arxiv_loader import ArxivDataLoader
from src.llm_client.get_client import get_client
from src.gnn.trainer import GNNTrainer
from src.util.util import load_config, setup_output_dir, parse_node_text
from src.util.log import setup_logger

def parse_args():
    parser = argparse.ArgumentParser(description="LLM+Graph TAGs Pipeline")
    parser.add_argument('-c', '--config', required=True, help="Path to config file")
    parser.add_argument('-e', '--experiment', type=str, help="Experiment name (overrides config)")
    parser.add_argument('--cache', type=str, help="Path to existing LLM prediction cache")
    return parser.parse_args()

def prepare_label_mapping(loader):
    """
    Prepare Label Mapping (String -> Index) for Pseudo-labeling.
    Normalize everything to uppercase 'CS.XX' for matching.
    """
    label_map_inv = {}
    if loader.label_map:
        for idx, cat_str in loader.label_map.items():
            # Example: 'arxiv cs na' -> 'CS.NA'
            clean_cat = cat_str.upper().replace('ARXIV ', '').replace(' ', '.')
            label_map_inv[clean_cat] = idx
    return label_map_inv

def run_llm_inference(config, loader, target_indices, output_dir, logger, cache_path=None):
    """
    Run LLM inference or load from cache.
    Returns a dictionary of predictions: {node_idx: {'llm_predict': str, 'llm_confident': float}}
    """
    logger.info("Starting LLM Prediction Phase...")
    
    # Priority 1: User specified cache
    # Priority 2: Default location in output_dir
    llm_cache_path = cache_path if cache_path else os.path.join(output_dir, "llm_predict.json")
    
    predictions = {}
    
    if os.path.exists(llm_cache_path):
        logger.info(f"Loading LLM predictions from cache: {llm_cache_path}")
        with open(llm_cache_path, 'r') as f:
            predictions = json.load(f)
            # Convert dictionary keys back to integers (JSON keys are strings)
            # Ensure compatibility with both old and new format (if any legacy cache exists)
            # New format uses: llm_predict, llm_confident
            # Old format used: category, confidence
            
            clean_preds = {}
            for k, v in predictions.items():
                node_idx = int(k)
                # Normalize keys
                cat = v.get('llm_predict', v.get('category'))
                conf = v.get('llm_confident', v.get('confidence'))
                clean_preds[node_idx] = {
                    "llm_predict": cat,
                    "llm_confident": conf
                }
            predictions = clean_preds
            
    else:
        logger.info(f"No cache found at {llm_cache_path}. Running LLM inference...")
        
        # If writing new cache, always save to the output_dir, even if user provided a custom path input
        # Reason: Not to overwrite external cache, but save current run's results locally.
        # But wait, if user provided a cache and it didn't exist, maybe they want to save there?
        # Usually user provides cache solely for READING shared results. 
        # Let's stick to saving in the current experiment output dir.
        save_path = os.path.join(output_dir, "llm_predict.json")
        
        # Setup Client
        llm_conf = config['llm']
        llm_conf['system_prompt'] = config['dataset']['system_prompt']
        client = get_client(llm_conf)
        candidates = config['dataset']['categories']
        prompt_template = config['dataset']['prompt_template']
        
        for idx in tqdm(target_indices, desc="LLM Inference"):
            raw_node_text = loader.get_formatted_message(idx)
            title, abstract = parse_node_text(raw_node_text)
            
            # Format prompt
            message = prompt_template.format(title=title, abstract=abstract)
            
            pred_cat, log_prob = client.predict(message=message, candidates=candidates)
            
            predictions[idx] = {
                "llm_predict": pred_cat,
                "llm_confident": log_prob
            }
            
        # Save Cache
        with open(save_path, 'w') as f:
            json.dump(predictions, f, indent=2)
        logger.info(f"LLM predictions saved to {save_path}")
        
    return predictions

def generate_pseudo_labels(predictions, label_map_inv, threshold, logger):
    """
    Filter predictions by confidence threshold and map to class indices.
    Returns lists of indices and labels.
    """
    logger.info("Filtering and Augmenting Training Set...")
    pseudo_labels = []
    pseudo_indices = []
    
    for idx, res in predictions.items():
        prob = math.exp(res['llm_confident'])
        
        if prob >= threshold:
            # Normalize prediction to match label_map_inv keys (e.g. 'CS.NA')
            pred_cat_norm = res['llm_predict'].upper()
            
            if pred_cat_norm in label_map_inv:
                cat_idx = label_map_inv[pred_cat_norm]
                pseudo_indices.append(idx)
                pseudo_labels.append(cat_idx)
            else:
                pass
                
    logger.info(f"Added {len(pseudo_indices)} pseudo-labels (Threshold: {threshold})")
    return pseudo_indices, pseudo_labels

def train_gnn_model(config, data, train_mask, num_classes, output_dir, logger):
    """
    Initialize and train the GNN model.
    Returns the trained trainer instance (model is inside).
    """
    logger.info("Starting GNN Training...")
    num_features = data.x.shape[1]
    
    gnn_config = config['gnn']
    trainer = GNNTrainer(gnn_config, num_features, num_classes)
    
    # Training Loop
    pbar = tqdm(range(trainer.epochs), desc="GNN Training")
    for epoch in pbar:
        loss = trainer.train(data, train_mask)
        pbar.set_postfix({'loss': f"{loss:.4f}"})
        
    # Save Model
    model_path = os.path.join(output_dir, "gnn_model.pt")
    trainer.save(model_path)
    logger.info(f"GNN Model saved to {model_path}")
    
    return trainer

def evaluate_and_save(trainer, data, split_idx, loader, predictions, pseudo_indices, output_dir, logger):
    """
    Run final inference, calculate metrics, and save results.
    """
    logger.info("Running Final Inference...")
    
    all_preds = trainer.predict(data) # [num_nodes] class indices
    
    # Calculate Accuracy on Test Set
    test_idx = split_idx['test']
    
    # Move to CPU for numpy operations/saving
    final_preds = all_preds.cpu().numpy()
    gt = data.y.squeeze().cpu().numpy()
    test_idx_np = test_idx.cpu().numpy()
    
    test_correct = (final_preds[test_idx_np] == gt[test_idx_np]).sum()
    total_count = len(test_idx_np)
    test_acc = test_correct / total_count
    
    logger.info(f"Final Test Accuracy: {test_acc:.4f}")
    
    # Save Results
    results = {
        "accuracy": float(test_acc * 100),
        "correct_count": int(test_correct),
        "total_count": int(total_count),
        "results": []
    }
    
    target_indices = split_idx['test'].tolist()
    
    for idx_val in target_indices:
        idx = int(idx_val)
        
        # Get LLM prediction info if available
        llm_res = predictions.get(idx, {})
        llm_cat = llm_res.get('category', None)
        llm_conf = llm_res.get('confidence', None)
        
        # Get GNN prediction
        gnn_pred_idx = int(final_preds[idx])
        
        # Helper to get clean category string
        def get_cat_str(cat_idx):
            if not loader.label_map:
                return str(cat_idx)
            s = loader.label_map.get(cat_idx, str(cat_idx))
            return s.replace('arxiv ', '') if isinstance(s, str) else s

        gnn_cat = get_cat_str(gnn_pred_idx)
        ground_truth_idx = int(gt[idx])
        ground_truth_cat = get_cat_str(ground_truth_idx)
        
        is_correct = bool(gnn_pred_idx == ground_truth_idx)

        results["results"].append({
            "node_idx": idx,
            "llm_predict": llm_cat,
            "llm_confident": llm_conf,
            "gnn_predict": gnn_cat,
            "final_predict": gnn_cat,
            "ground_truth": ground_truth_cat,
            "is_correct": is_correct
        })
        
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
        
    logger.info(f"Results saved to {results_path}")

def main():
    args = parse_args()
    config = load_config(args.config)
    
    exp_name = args.experiment if args.experiment else config['experiment']['name']
    
    output_dir = setup_output_dir(config, exp_name, args.config)
    logger, log_file = setup_logger(output_dir)
    
    logger.info(f"Starting Experiment: {exp_name}")
    logger.info(f"Output Directory: {output_dir}")
    
    # 1. Data Loading
    logger.info("Loading Data...")
    loader = ArxivDataLoader(root=config['dataset']['root'])
    data = loader.get_data()
    
    # 2. LLM Prediction
    # Use the loader's built-in inverse mapping logic (Standardized 'CS.XX' -> Index)
    label_map_inv = loader.get_inv_label_map()
    
    # Determine target nodes (Test Set)
    split_idx = loader.split_idx
    target_indices = split_idx['test'].tolist()
    
    predictions = run_llm_inference(config, loader, target_indices, output_dir, logger, cache_path=args.cache)

    # 3. Filtering and Augmentation
    threshold = config['pipeline']['confidence_threshold']
    pseudo_indices, pseudo_labels = generate_pseudo_labels(predictions, label_map_inv, threshold, logger)
    
    # Prepare Augmented Data for GNN Training
    train_idx = split_idx['train']
    new_train_idx = torch.cat([
        train_idx,
        torch.tensor(pseudo_indices, dtype=torch.long)
    ])
    
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[new_train_idx] = True
    
    # Backup original labels
    original_y = data.y.clone()
    
    # Inject pseudo-labels
    if pseudo_indices:
        pseudo_tensor = torch.tensor(pseudo_labels, dtype=torch.long).unsqueeze(1)
        data.y[pseudo_indices] = pseudo_tensor
    
    # 4. GNN Training
    trainer = train_gnn_model(config, data, train_mask, loader.dataset.num_classes, output_dir, logger)
    
    # 5. Inference & Evaluation
    # Restore original labels for correct evaluation
    data.y = original_y
    evaluate_and_save(trainer, data, split_idx, loader, predictions, pseudo_indices, output_dir, logger)

    logger.info("Pipeline Completed.")

if __name__ == "__main__":
    main()
