import argparse
import json
import math
import os
from typing import Any, Dict, Optional

import numpy as np
import torch

from src.data_loader.get_dataset import get_dataset
from src.util.log import setup_logger
from src.util.util import load_config, setup_output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="Final fusion pipeline")
    parser.add_argument("-c", "--config", required=True, help="Path to config file")
    return parser.parse_args()


def load_predictions(path: str) -> Dict[int, Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def normalize_llm_label(label: Any) -> str:
    if label is None:
        return ""
    return str(label).strip().upper().replace("ARXIV ", "").replace(" ", ".")


def to_probability(conf_value: Any, pred_label: Optional[str] = None) -> float:
    """
    Convert confidence to probability in [0, 1].
    Supports scalar values and dict[label -> logprob/probability].
    """
    val = None

    if isinstance(conf_value, dict):
        if pred_label and pred_label in conf_value:
            val = conf_value[pred_label]
        elif pred_label and pred_label.lower() in conf_value:
            val = conf_value[pred_label.lower()]
        elif conf_value:
            val = max(conf_value.values())
    else:
        val = conf_value

    try:
        val = float(val)
    except (TypeError, ValueError):
        return 0.0

    prob = math.exp(val) if val < 0 else val
    if prob < 0:
        return 0.0
    if prob > 1.0:
        return 1.0
    return prob


def get_category_str(label_map: Optional[Dict[int, str]], idx: Optional[int]) -> Optional[str]:
    if idx is None:
        return None
    if label_map is None:
        return str(idx)
    return str(label_map.get(int(idx), str(idx))).replace("arxiv ", "")


def safe_split_indices(indices: Any) -> np.ndarray:
    if torch.is_tensor(indices):
        return indices.cpu().numpy()
    return np.array(indices)


def main():
    args = parse_args()
    config = load_config(args.config)

    exp_name = config["experiment"]["name"]
    output_dir = setup_output_dir(config, exp_name, args.config)

    logger, _ = setup_logger(output_dir, log_file_name="run_final.log")
    logger.info(f"Starting final fusion for experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")

    final_cfg = config.get("final_prediction", {})
    method = final_cfg.get("method")
    if method not in {"llm_only", "gnn_only", "hard_threshold"}:
        raise ValueError(
            "Unsupported final_prediction.method. "
            f"Got: {method}. Expected one of: llm_only, gnn_only, hard_threshold"
        )

    threshold = float(final_cfg.get("threshold", 0.99))
    output_tag = str(final_cfg.get("output_file", method)).strip() or method
    safe_output_tag = output_tag.replace(" ", "_").replace("/", "_").replace("\\", "_")
    logger.info(f"Final method: {method}")
    logger.info(f"Output tag: {safe_output_tag}")
    if method == "hard_threshold":
        logger.info(f"Hard threshold: {threshold}")

    llm_path = os.path.join(output_dir, "llm_predictions.json")
    gnn_path = os.path.join(output_dir, "gnn_predictions.json")

    if not os.path.exists(llm_path):
        raise FileNotFoundError(f"Missing required file: {llm_path}")
    if not os.path.exists(gnn_path):
        raise FileNotFoundError(f"Missing required file: {gnn_path}")

    logger.info("Loading dataset and predictions...")
    loader = get_dataset(config)
    data = loader.get_data()
    split_idx = loader.split_idx
    inv_map = loader.get_inv_label_map()
    label_map = loader.label_map

    llm_preds = load_predictions(llm_path)
    gnn_preds = load_predictions(gnn_path)

    gt = data.y.squeeze().cpu().numpy()

    node_to_subset = {}
    for split_name, indices in split_idx.items():
        for idx in safe_split_indices(indices):
            node_to_subset[int(idx)] = split_name

    split_names = list(split_idx.keys()) + ["unknown"]
    split_stats = {
        name: {"correct": 0, "total": 0, "resolved": 0} for name in split_names
    }

    source_counts = {"llm": 0, "gnn": 0, "none": 0}
    predictions_out = []

    correct_total = 0
    resolved_total = 0
    total_nodes = data.num_nodes

    logger.info("Running final fusion for all nodes...")
    for node_idx in range(total_nodes):
        subset = node_to_subset.get(node_idx, "unknown")
        gt_idx = int(gt[node_idx])

        llm_res = llm_preds.get(node_idx, {})
        gnn_res = gnn_preds.get(node_idx, {})

        llm_label_raw = llm_res.get("llm_predict")
        llm_label_norm = normalize_llm_label(llm_label_raw)
        llm_idx = inv_map.get(llm_label_norm)

        llm_conf_raw = llm_res.get("llm_confident", -999)
        llm_prob = to_probability(llm_conf_raw, llm_label_norm)

        gnn_idx = gnn_res.get("gnn_predict")
        if gnn_idx is not None:
            gnn_idx = int(gnn_idx)

        final_idx = None
        source = "none"

        if method == "llm_only":
            if llm_idx is not None:
                final_idx = int(llm_idx)
                source = "llm"
        elif method == "gnn_only":
            if gnn_idx is not None:
                final_idx = int(gnn_idx)
                source = "gnn"
        else:  # hard_threshold
            if llm_idx is not None and llm_prob >= threshold:
                final_idx = int(llm_idx)
                source = "llm"
            elif gnn_idx is not None:
                final_idx = int(gnn_idx)
                source = "gnn"
            elif llm_idx is not None:
                # Fallback when GNN prediction is unexpectedly missing
                final_idx = int(llm_idx)
                source = "llm"

        source_counts[source] += 1

        is_correct = False
        if final_idx is not None:
            resolved_total += 1
            is_correct = final_idx == gt_idx
            if is_correct:
                correct_total += 1

        split_stats[subset]["total"] += 1
        if final_idx is not None:
            split_stats[subset]["resolved"] += 1
        if is_correct:
            split_stats[subset]["correct"] += 1

        predictions_out.append(
            {
                "node_idx": node_idx,
                "subset": subset,
                "llm_predict": llm_label_raw,
                "llm_confident": llm_conf_raw,
                "llm_confidence_prob": llm_prob,
                "gnn_predict": gnn_idx,
                "final_source": source,
                "final_predict": final_idx,
                "ground_truth": gt_idx,
                "is_correct": is_correct,
                "final_predict_category": get_category_str(label_map, final_idx),
                "ground_truth_category": get_category_str(label_map, gt_idx),
            }
        )

    overall_acc = (correct_total / total_nodes * 100.0) if total_nodes > 0 else 0.0
    resolved_acc = (correct_total / resolved_total * 100.0) if resolved_total > 0 else 0.0

    metrics: Dict[str, Any] = {
        "method": method,
        "overall": {
            "accuracy": overall_acc,
            "correct": correct_total,
            "total": total_nodes,
            "resolved": resolved_total,
            "resolved_accuracy": resolved_acc,
        },
        "source_counts": source_counts,
        "splits": {},
    }

    for split_name, st in split_stats.items():
        if st["total"] == 0:
            continue
        split_acc = st["correct"] / st["total"] * 100.0
        split_resolved_acc = st["correct"] / st["resolved"] * 100.0 if st["resolved"] > 0 else 0.0
        metrics["splits"][split_name] = {
            "accuracy": split_acc,
            "correct": st["correct"],
            "total": st["total"],
            "resolved": st["resolved"],
            "resolved_accuracy": split_resolved_acc,
        }

    if method == "hard_threshold":
        metrics["threshold"] = threshold

    metrics_path = os.path.join(output_dir, f"final_metrics.{safe_output_tag}.json")
    preds_path = os.path.join(output_dir, f"final_predictions.{safe_output_tag}.json")

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with open(preds_path, "w", encoding="utf-8") as f:
        json.dump(predictions_out, f, indent=2)

    logger.info(
        f"[OVERALL] Acc={overall_acc:.4f} | Resolved={resolved_total}/{total_nodes} | "
        f"Resolved Acc={resolved_acc:.4f}"
    )
    logger.info(f"Saved metrics to {metrics_path}")
    logger.info(f"Saved node predictions to {preds_path}")


if __name__ == "__main__":
    main()
