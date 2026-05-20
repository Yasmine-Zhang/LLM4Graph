import argparse
import os
import json
import torch
import math
from src.gnn.trainer import GNNTrainer
from src.data_loader.get_dataset import get_dataset
from src.data_selector.selector import select_anchors
from src.util.util import load_config, setup_output_dir
from src.util.log import setup_logger


def parse_args():
    parser = argparse.ArgumentParser(description="GNN-only pipeline (filter/train/infer)")
    parser.add_argument('-c', '--config', required=True, help="Path to config file")
    parser.add_argument('-e', '--experiment', type=str, help="Experiment name (overrides config)")
    return parser.parse_args()


def train_gnn_model(config, data, train_mask, num_classes, output_dir, logger, split_idx=None, original_y=None):
    """
    Initialize and train the GNN model with model selection (Early Stopping).
    Returns the trained trainer instance (model is inside).
    """
    logger.info("Starting GNN Training...")
    num_features = data.x.shape[1]
    
    gnn_config = config['gnn']
    trainer = GNNTrainer(gnn_config, num_features, num_classes)
    
    # Validation Setup
    valid_mask = None
    if split_idx is not None and 'valid' in split_idx:
        valid_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        valid_mask[split_idx['valid']] = True
        logger.info("Validation set enabled for model selection.")

    model_path = os.path.join(output_dir, "gnn_model.pt")
    best_val_acc = 0.0
    best_epoch = -1
    
    # Training Loop
    from tqdm import tqdm
    pbar = tqdm(range(trainer.epochs), desc="GNN Training")
    for epoch in pbar:
        loss = trainer.train(data, train_mask)
        
        current_val_acc = 0.0
        if valid_mask is not None:
            # Use original_y as ground truth for validation
            current_val_acc = trainer.evaluate(data, valid_mask, true_y=original_y)
            
            # Save Best Model
            if current_val_acc > best_val_acc:
                best_val_acc = current_val_acc
                best_epoch = epoch
                trainer.save(model_path)
            
            pbar.set_postfix({'loss': f"{loss:.4f}", 'val_acc': f"{current_val_acc:.4f}"})
        else:
            pbar.set_postfix({'loss': f"{loss:.4f}"})
            # If no validation, just save the last one
            trainer.save(model_path)
        
        # Log to file every 50 epochs
        if (epoch + 1) % 50 == 0:
            log_msg = f"Epoch {epoch+1:03d}/{trainer.epochs} | Loss: {loss:.4f}"
            if valid_mask is not None:
                log_msg += f" | Val Acc: {current_val_acc:.4f} (Best: {best_val_acc:.4f} at Ep {best_epoch})"
            logger.info(log_msg)
        
    if valid_mask is not None:
        logger.info(f"Training Finished. Best Validation Epoch: {best_epoch} (Acc: {best_val_acc:.4f})")
        logger.info(f"Loading best model from {model_path}")
        trainer.load(model_path)
    else:
        logger.info(f"Training Finished. GNN Model saved to {model_path}")
    
    return trainer


def main():
    args = parse_args()
    config = load_config(args.config)
    
    exp_name = args.experiment if args.experiment else config['experiment']['name']
    output_dir = os.path.join("output", exp_name)
    
    os.makedirs(output_dir, exist_ok=True)
    
    logger, log_file = setup_logger(output_dir, log_file_name="run_gnn.log")
    
    logger.info(f"Starting GNN Pipeline for Experiment: {exp_name}")
    logger.info(f"Output Directory: {output_dir}")
    
    # 1. Load Data
    logger.info("Loading Data...")
    loader = get_dataset(config)
    data = loader.get_data()
    split_idx = loader.split_idx
    
    # 2. Check if this is supervised (ground truth) ablation
    selector_config = config.get('gnn_training_data_filter', {})
    is_supervised = selector_config.get('method') == 'supervised'
    
    # Load LLM Predictions (or skip if supervised ablation)
    predictions = {}
    if not is_supervised:
        llm_path = os.path.join(output_dir, "llm_predictions.json")
        if not os.path.exists(llm_path):
            raise FileNotFoundError(f"LLM predictions not found at {llm_path}. Please run run_llm.py first.")
        
        logger.info(f"Loading LLM predictions from {llm_path}")
        with open(llm_path, 'r') as f:
            raw_llm = json.load(f)
        
        predictions = {int(k): v for k, v in raw_llm.items()}
        logger.info(f"Loaded {len(predictions)} LLM predictions")
    else:
        logger.info("Supervised ablation mode: Using ground truth labels instead of LLM predictions")
    
    # Get label mapping
    label_map_inv = loader.get_inv_label_map()
    
    # 3. Filter and Augment Training Set
    logger.info("Filtering and Augmenting Training Set...")
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    original_y = data.y.clone()
    
    # Extract selector config (already retrieved above)
    if not selector_config:
        logger.warning("No 'gnn_training_data_filter' config found! Using default (confidence_threshold, threshold=0.9)")
        selector_config = {'method': 'confidence_threshold', 'threshold': 0.9}
    
    # Call Selector (Pass split_idx and gt_y for supervised strategy)
    pseudo_indices, pseudo_labels = select_anchors(
        predictions, label_map_inv, selector_config, logger,
        split_idx=split_idx, gt_y=data.y
    )
    
    if not pseudo_indices:
        logger.warning("No pseudo-labels generated! GNN cannot train.")
        return
    
    logger.info(f"Selected {len(pseudo_indices)} pseudo-labeled nodes for training")
    
    new_train_idx = torch.tensor(pseudo_indices, dtype=torch.long)
    train_mask[new_train_idx] = True
    
    # Inject pseudo-labels
    if pseudo_indices:
        pseudo_tensor = torch.tensor(pseudo_labels, dtype=torch.long).unsqueeze(1)
        data.y[pseudo_indices] = pseudo_tensor
    
    # 4. Train GNN
    if 'gnn' not in config:
        logger.error("'gnn' config missing! Cannot train GNN.")
        return
    
    trainer = train_gnn_model(
        config, data, train_mask, loader.dataset.num_classes,
        output_dir, logger, split_idx=split_idx, original_y=original_y
    )
    
    # 5. GNN Inference with Full Softmax Probabilities
    logger.info("Running GNN Inference with full probability distribution...")
    trainer.run_gnn_inference_with_probs(data, output_dir, logger, loader.dataset.num_classes)
    
    # Restore original labels
    data.y = original_y
    
    logger.info("GNN Pipeline Completed.")


if __name__ == "__main__":
    main()
