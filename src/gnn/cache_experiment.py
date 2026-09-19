import argparse
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch_geometric
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from src.gnn.trainer import GNNTrainer


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def load_annotations(path, label_map, num_nodes, paper_ids=None):
    with Path(path).open(encoding="utf-8") as stream:
        records = json.load(stream, object_pairs_hook=unique_object)
    if set(records) != {str(node) for node in range(num_nodes)}:
        raise ValueError("Cache must cover every node exactly once with canonical integer keys")
    labels = np.empty(num_nodes, dtype=np.int64)
    confidence = np.full(num_nodes, np.nan, dtype=np.float64)
    models, protocols = set(), set()
    for node in range(num_nodes):
        record = records[str(node)]
        if record.get("status") != "success" or record.get("node_id") != node:
            raise ValueError(f"Invalid status or node_id at node {node}")
        category = record.get("llm_predict")
        if not isinstance(category, str) or category.upper() not in label_map:
            raise ValueError(f"Invalid category at node {node}")
        if paper_ids is not None and str(record.get("paper_id")) != str(paper_ids[node]):
            raise ValueError(f"Paper alignment mismatch at node {node}")
        labels[node] = label_map[category.upper()]
        score = record.get("self_reported_confidence")
        if score is not None:
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"Invalid self-reported confidence at node {node}")
            confidence[node] = score
        models.add(record.get("model"))
        protocols.add(record.get("protocol_id"))
    if len(models) != 1 or None in models or len(protocols) != 1 or None in protocols:
        raise ValueError("Each experiment must contain one teacher and one annotation protocol")
    return labels, confidence, {"model": next(iter(models)), "protocol_id": next(iter(protocols))}


def select_mask(confidence, method, threshold):
    if method == "all":
        return np.ones(len(confidence), dtype=bool)
    if method != "confidence_threshold" or not 0 <= threshold <= 1:
        raise ValueError("Invalid anchor selection")
    mask = np.isfinite(confidence) & (confidence >= threshold)
    if not mask.any():
        raise ValueError("No anchors satisfy the confidence threshold")
    return mask


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_graph(root, adjacency_format="coo"):
    root = Path(root)
    raw = root / "raw"
    num_nodes = int(pd.read_csv(raw / "num-node-list.csv.gz", header=None).iloc[0, 0])
    num_edges = int(pd.read_csv(raw / "num-edge-list.csv.gz", header=None).iloc[0, 0])
    features = pd.read_csv(raw / "node-feat.csv.gz", header=None, dtype=np.float32).to_numpy()
    edges = pd.read_csv(raw / "edge.csv.gz", header=None, dtype=np.int64).to_numpy()
    if features.shape != (num_nodes, 128) or not np.isfinite(features).all():
        raise ValueError("Invalid or incomplete Arxiv features")
    if edges.shape != (num_edges, 2) or edges.min() < 0 or edges.max() >= num_nodes:
        raise ValueError("Invalid or incomplete Arxiv edges")
    mapping = pd.read_csv(root / "mapping" / "labelidx2arxivcategeory.csv.gz")
    label_map = {
        category.upper().replace("ARXIV ", "").replace(" ", "."): int(label)
        for label, category in zip(mapping["label idx"], mapping["arxiv category"])
    }
    if len(label_map) != 40 or set(label_map.values()) != set(range(40)):
        raise ValueError("Expected the official 40-class mapping")
    paper_map = pd.read_csv(root / "mapping" / "nodeidx2paperid.csv.gz").sort_values("node idx")
    if not np.array_equal(paper_map["node idx"].to_numpy(), np.arange(num_nodes)):
        raise ValueError("Incomplete node-to-paper mapping")
    edge_index = to_undirected(torch.from_numpy(edges.T.copy()), num_nodes=num_nodes)
    if adjacency_format == "csr":
        edge_index = torch.sparse_coo_tensor(
            edge_index.flip(0), torch.ones(edge_index.shape[1]), (num_nodes, num_nodes)
        ).coalesce().to_sparse_csr()
    elif adjacency_format != "coo":
        raise ValueError("Unknown adjacency format")
    return Data(x=torch.from_numpy(features), edge_index=edge_index, num_nodes=num_nodes), label_map, paper_map["paper id"].to_numpy()


def run(args):
    if args.epochs < 1 or args.num_layers < 2 or args.hidden_channels < 1 or not 0 <= args.dropout < 1 or args.lr <= 0:
        raise ValueError("Invalid training hyperparameters; the existing GCN requires at least two layers")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        raise ValueError("Seeds and model types must be unique")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; CPU fallback must be explicit")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite experiment: {args.output}")
    torch.set_num_threads(args.threads)
    adjacency_format = getattr(args, "adjacency_format", "coo")
    graph, label_map, paper_ids = load_graph(args.data_root, adjacency_format)
    labels, confidence, teacher = load_annotations(args.cache, label_map, graph.num_nodes, paper_ids)
    anchor_mask = select_mask(confidence, args.selection, args.threshold)
    args.output.mkdir(parents=True)
    np.savez_compressed(args.output / "annotations.npz", labels=labels, confidence=confidence, anchor_mask=anchor_mask)
    degree = (torch.diff(graph.edge_index.crow_indices()).numpy() if adjacency_format == "csr"
              else torch.bincount(graph.edge_index[0], minlength=graph.num_nodes).numpy())
    np.save(args.output / "degree.npy", degree)
    training_config = {
        "hidden_channels": args.hidden_channels, "num_layers": args.num_layers,
        "dropout": args.dropout, "lr": args.lr, "epochs": args.epochs,
        "weight_decay": 0.0, "device": args.device,
    }
    source_files = [Path(__file__), Path(__file__).with_name("model.py"), Path(__file__).with_name("trainer.py")]
    input_files = [args.cache] + [args.data_root / relative for relative in (
        "raw/node-feat.csv.gz", "raw/edge.csv.gz", "raw/num-node-list.csv.gz", "raw/num-edge-list.csv.gz",
        "mapping/labelidx2arxivcategeory.csv.gz", "mapping/nodeidx2paperid.csv.gz",
    )]
    protocol = {
        "teacher": teacher, "cache": str(args.cache.resolve()), "data_root": str(args.data_root.resolve()),
        "training": training_config, "models": args.models, "seeds": args.seeds,
        "selection": args.selection, "threshold": args.threshold,
        "confidence_kind": "self_reported_not_calibrated", "num_nodes": graph.num_nodes,
        "num_anchors": int(anchor_mask.sum()), "class_mapping": label_map,
        "anchor_class_counts": np.bincount(labels[anchor_mask], minlength=40).tolist(),
        "selection_uses_ground_truth": False, "checkpoint": "fixed_final_epoch",
        "features": "official_ogbn_arxiv_128d", "graph": "undirected_coalesced", "adjacency_format": adjacency_format,
        "transductive_batchnorm": True, "bitwise_cuda_reproducibility_guaranteed": False,
        "versions": {"python": sys.version, "torch": torch.__version__, "numpy": np.__version__, "pyg": torch_geometric.__version__},
        "device_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else "cpu",
        "source_hashes": {str(path.resolve()): sha256(path) for path in source_files},
        "input_hashes": {str(path.resolve()): sha256(path) for path in input_files},
    }
    (args.output / "protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    pseudo_targets = np.full(graph.num_nodes, -1, dtype=np.int64)
    pseudo_targets[anchor_mask] = labels[anchor_mask]
    graph.y = torch.from_numpy(pseudo_targets)
    graph = graph.to(args.device)
    mask = torch.from_numpy(anchor_mask).to(args.device)
    print(f"Teacher={teacher['model']} nodes={graph.num_nodes} anchors={int(mask.sum())} device={args.device}", flush=True)
    for seed in args.seeds:
        for model_type in args.models:
            destination = args.output / f"{model_type}.seed{seed}"
            destination.mkdir()
            seed_everything(seed)
            trainer = GNNTrainer(dict(training_config, model_type=model_type), graph.num_features, 40)
            started = time.perf_counter()
            with (destination / "loss.jsonl").open("w", encoding="utf-8") as stream:
                for epoch in range(args.epochs):
                    loss = trainer.train(graph, mask)
                    if not math.isfinite(loss):
                        raise RuntimeError(f"Nonfinite loss: {model_type}, epoch {epoch + 1}")
                    stream.write(json.dumps({"epoch": epoch + 1, "loss": loss}) + "\n")
                    if epoch == 0 or (epoch + 1) % 25 == 0 or epoch + 1 == args.epochs:
                        stream.flush()
                        print(f"{model_type} seed={seed} epoch={epoch + 1}/{args.epochs} loss={loss:.6f}", flush=True)
            logits = trainer.get_probs(graph).cpu().numpy()
            if not np.isfinite(logits).all():
                raise RuntimeError("Nonfinite final logits")
            trainer.save(destination / "model.pt")
            np.savez_compressed(destination / "predictions.npz", logits=logits, predictions=logits.argmax(axis=1))
            completed = {"seconds": time.perf_counter() - started, "epochs": args.epochs,
                         "parameters": sum(parameter.numel() for parameter in trainer.model.parameters()),
                         "prediction_sha256": sha256(destination / "predictions.npz")}
            (destination / "completed.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
            del trainer
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
    (args.output / "completed.json").write_text(json.dumps({"training_complete": True}), encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="Cache-only, fixed-epoch GCN/MLP experiment; never reads ground truth")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/ogbn_arxiv"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", nargs="+", choices=["gcn", "mlp"], default=["gcn", "mlp"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--selection", choices=["all", "confidence_threshold"], default="confidence_threshold")
    parser.add_argument("--threshold", type=float, default=0.9)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--hidden-channels", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--adjacency-format", choices=["coo", "csr"], default="coo")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())