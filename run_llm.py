
import argparse
import os
import json
import torch
import math
import shutil
import numpy as np
from tqdm import tqdm
from multiprocessing import Process
from src.llm_client.get_client import get_client
from src.data_loader.get_dataset import get_dataset
from src.util.util import load_config, setup_output_dir
from src.util.log import setup_logger
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="LLM-only parallel inference pipeline")
    parser.add_argument('-c', '--config', required=True, help="Path to config file")
    parser.add_argument('-l', '--llm_cache', type=str, help="Path to existing LLM prediction cache")
    parser.add_argument('--num_shards', type=int, default=1, help="Number of parallel shards")
    # No --merge_only, always merge after all shards
    return parser.parse_args()



def run_shard(shard_id, num_shards, args):
    config = load_config(args.config)
    exp_name = config['experiment']['name']
    output_dir = setup_output_dir(config, exp_name, args.config)
    logger, log_file = setup_logger(output_dir, log_file_name=f"run_shard{shard_id}.log")

    logger.info(f"Shard {shard_id}: Loading Data...")
    loader = get_dataset(config)
    data = loader.get_data()
    all_indices = list(range(data.num_nodes))
    chunk_size = int(math.ceil(len(all_indices) / num_shards))
    start_idx = shard_id * chunk_size
    end_idx = min((shard_id + 1) * chunk_size, len(all_indices))
    target_indices = all_indices[start_idx:end_idx]
    logger.info(f"Shard {shard_id}: Nodes {start_idx}-{end_idx} ({len(target_indices)} total)")

    llm_conf = config['llm']
    if 'system_prompt' not in config['dataset'] or not config['dataset']['system_prompt']:
        llm_conf['system_prompt'] = loader.get_system_prompt()
    else:
        llm_conf['system_prompt'] = config['dataset']['system_prompt']

    # For TransformersClient: assign device_id based on shard_id (round-robin across available GPUs)
    # This enables true parallel inference on multiple GPUs
    if llm_conf.get('type') == 'TransformersClient':
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
        assigned_device_id = shard_id % num_gpus
        llm_conf['device_id'] = assigned_device_id
        logger.info(f"Shard {shard_id}: Assigned to GPU {assigned_device_id} (out of {num_gpus})")

    # Per-shard output file
    shard_output = os.path.join(output_dir, f"llm_predictions.{shard_id}.json")
    cache_hit = False
    predictions = {}
    if os.path.exists(shard_output):
        try:
            with open(shard_output, 'r') as f:
                loaded_preds = json.load(f)
            for k, v in loaded_preds.items():
                node_idx = int(k)
                predictions[node_idx] = v
            logger.info(f"Shard {shard_id}: Loaded {len(predictions)} predictions. Resuming...")
            cache_hit = True if len(predictions) == len(target_indices) else False
        except Exception as e:
            logger.warning(f"Shard {shard_id}: Failed to read cache: {e}")

    if not cache_hit:
        client = get_client(llm_conf)
        predictions = client.run_inference(
            loader=loader,
            target_indices=target_indices,
            output_dir=output_dir,
            logger=logger,
            prompt_template=config['dataset']['prompt_template'],
            candidates=config['dataset']['categories'],
            llm_cache=shard_output
        )
        # Save after run (should already be saved periodically)
        with open(shard_output, 'w') as f:
            json.dump(predictions, f, indent=2)
    logger.info(f"Shard {shard_id}: Finished. Predictions saved to {shard_output}")

def merge_shard_outputs(output_dir, num_shards):
    merged = {}
    for i in range(num_shards):
        shard_file = os.path.join(output_dir, f"llm_predictions.{i}.json")
        if os.path.exists(shard_file):
            with open(shard_file, 'r') as f:
                d = json.load(f)
                merged.update({int(k): v for k, v in d.items()})
    merged_file = os.path.join(output_dir, "llm_predictions.json")
    with open(merged_file, 'w') as f:
        json.dump(merged, f, indent=2)
    print(f"Merged {num_shards} shards into {merged_file} ({len(merged)} nodes)")

def main():
    args = parse_args()
    config = load_config(args.config)
    exp_name = config['experiment']['name']
    output_dir = setup_output_dir(config, exp_name, args.config)

    # Launch parallel processes for each shard
    procs = []
    for shard_id in range(args.num_shards):
        p = Process(target=run_shard, args=(shard_id, args.num_shards, args))
        p.start()
        procs.append(p)
    shard_failures = []
    for i, p in enumerate(procs):
        p.join()
        if p.exitcode != 0:
            shard_failures.append((i, p.exitcode))

    if shard_failures:
        detail = ", ".join([f"shard {sid} exit={code}" for sid, code in shard_failures])
        raise RuntimeError(f"run_llm failed: {detail}")

    # Always merge outputs after all shards finish
    merge_shard_outputs(output_dir, args.num_shards)
    print("All shards complete. Merged output ready.")

if __name__ == "__main__":
    main()
