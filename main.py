import argparse
import os
import json
import torch
import math
import shutil
import numpy as np
from tqdm import tqdm
from src.llm_client.get_client import get_client
from src.data_loader.get_dataset import get_dataset
from src.gnn.trainer import GNNTrainer
from src.analysis.evaluator import evaluate_results
from src.util.util import load_config, setup_output_dir
from src.util.log import setup_logger
from src.data_selector.selector import select_anchors

def parse_args():
    parser = argparse.ArgumentParser(description="LLM+Graph TAGs Pipeline")
    parser.add_argument('-c', '--config', required=True, help="Path to config file")
    parser.add_argument('-e', '--experiment', type=str, help="Experiment name (overrides config)")
    parser.add_argument('-l', '--llm_cache', type=str, help="Path to existing LLM prediction cache")
    parser.add_argument('--shard_id', type=int, default=0, help="Shard ID for distributed inference")
    parser.add_argument('--num_shards', type=int, default=1, help="Total number of shards/GPUs")
    return parser.parse_args()

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
        
        # Log to file every 10 epochs
        if (epoch + 1) % 10 == 0:
            logger.info(f"Epoch {epoch+1:03d}/{trainer.epochs} | Loss: {loss:.4f}")
        
    # Save Model
    model_path = os.path.join(output_dir, "gnn_model.pt")
    trainer.save(model_path)
    logger.info(f"GNN Model saved to {model_path}")
    
    return trainer

def main():
    args = parse_args()
    config = load_config(args.config)
    
    exp_name = args.experiment if args.experiment else config['experiment']['name']
    
    output_dir = setup_output_dir(config, exp_name, args.config)
    logger, log_file = setup_logger(output_dir)
    
    logger.info(f"Starting Experiment: {exp_name}")
    logger.info(f"Output Directory: {output_dir}")
    
    # Handle LLM Cache Copying
    if args.llm_cache:
        if os.path.exists(args.llm_cache):
            target_cache_path = os.path.join(output_dir, "llm_predict.json")
            if not os.path.exists(target_cache_path): # Don't overwrite if already exists in output? Or should we?
                # User specifically asked to use this cache, so we should probably overwrite or ensure it's there.
                # But let's trigger the copy.
                shutil.copy(args.llm_cache, target_cache_path)
                logger.info(f"Copied LLM Cache from {args.llm_cache} to {target_cache_path}")
            else:
                 logger.warning(f"Target cache {target_cache_path} already exists. Using existing file instead of {args.llm_cache}")
        else:
            logger.error(f"Provided LLM cache path {args.llm_cache} does not exist!")

    # 1. Data Loading
    logger.info("Loading Data...")
    loader = get_dataset(config)
    data = loader.get_data()
    
    # 2. LLM Prediction
    # Use the loader's built-in inverse mapping logic (Standardized 'CS.XX' -> Index)
    label_map_inv = loader.get_inv_label_map()
    
    # Determine target nodes (Label-Free: Full Graph)
    # We predict logic for ALL nodes to find high-confidence anchors anywhere.
    split_idx = loader.split_idx
    all_indices = list(range(data.num_nodes))
    
    # Sharding for multi-gpu/parallel inference
    if args.num_shards > 1:
        chunk_size = int(math.ceil(len(all_indices) / args.num_shards))
        start_idx = args.shard_id * chunk_size
        end_idx = min((args.shard_id + 1) * chunk_size, len(all_indices))
        target_indices = all_indices[start_idx:end_idx]
        logger.info(f"Running in Sharded Mode: Shard {args.shard_id}/{args.num_shards} | Nodes {start_idx}-{end_idx} ({len(target_indices)} total)")
    else:
        target_indices = all_indices
    
    # Setup Client
    if 'llm' in config:
        llm_conf = config['llm']
        # Priority: Config > Loader Default
        if 'system_prompt' not in config['dataset'] or not config['dataset']['system_prompt']:
             llm_conf['system_prompt'] = loader.get_system_prompt()
        else:
             llm_conf['system_prompt'] = config['dataset']['system_prompt']
             
        client = get_client(llm_conf)
        
        predictions = client.run_inference(
            loader=loader, 
            target_indices=target_indices, 
            output_dir=output_dir, 
            logger=logger,
            prompt_template=config['dataset']['prompt_template'],
            candidates=config['dataset']['categories'],
            llm_cache=None # modified to use the copied file in output_dir
        )
    else:
        logger.info("[INFO] 'llm' config missing. Skipping LLM inference.")
        predictions = {}

    # 3. Filtering and Augmentation
    logger.info("Filtering and Augmenting Training Set...")
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    original_y = data.y.clone()
    
    # Extract selector config directly
    selector_config = config.get('data_filter', {})
    if not selector_config:
        logger.warning("No 'data_filter' config found! Using defaults or failing.")

    # Call Selector (Pass split_idx and gt_y for supervised strategy)
    pseudo_indices, pseudo_labels = select_anchors(
        predictions, label_map_inv, selector_config, logger, 
        split_idx=split_idx, gt_y=data.y
    )
    
    if not pseudo_indices:
        logger.warning("No pseudo-labels generated! GNN cannot train.")
        return

    new_train_idx = torch.tensor(pseudo_indices, dtype=torch.long)
    train_mask[new_train_idx] = True
    
    # Inject pseudo-labels (Override data.y for GNN training)
    # Even if supervised, we do this to ensure data.y matches what GNN expects at train_mask
    if pseudo_indices:
        pseudo_tensor = torch.tensor(pseudo_labels, dtype=torch.long).unsqueeze(1)
        data.y[pseudo_indices] = pseudo_tensor
    
    # 4. GNN Training
    trainer = train_gnn_model(config, data, train_mask, loader.dataset.num_classes, output_dir, logger)
    
    # 5. GNN 
    # Decoupled inference step, saves to gnn_predict.json
    trainer.run_gnn_inference(data, output_dir, logger)

    # 6. Evaluation & Analysis
    # Decoupled analysis step, reads json files and produces results.json
    # Restore original labels for correct evaluation
    data.y = original_y
    evaluate_results(data, split_idx, loader, output_dir, logger, config)

    logger.info("Pipeline Completed.")

if __name__ == "__main__":
    main()
