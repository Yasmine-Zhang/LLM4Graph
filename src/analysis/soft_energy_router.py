import math
from typing import Any, Dict

import numpy as np


def _normalize_llm_label(label: Any) -> str:
    if label is None:
        return ""
    return str(label).strip().upper().replace("ARXIV ", "").replace(" ", ".")


def _value_to_probability(value: Any) -> float:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return 0.0

    prob = math.exp(val) if val < 0 else val
    if prob < 0.0:
        return 0.0
    if prob > 1.0:
        return 1.0
    return prob


def _build_llm_probs(
    llm_res: Dict[str, Any],
    inv_map: Dict[str, int],
    num_classes: int,
) -> Dict[str, Any]:
    probs = np.zeros(num_classes, dtype=np.float64)

    llm_label_raw = llm_res.get("llm_predict")
    llm_label_norm = _normalize_llm_label(llm_label_raw)
    llm_idx = inv_map.get(llm_label_norm, -1)

    conf_raw = llm_res.get("llm_confident", {})
    if isinstance(conf_raw, dict):
        for k, v in conf_raw.items():
            idx = inv_map.get(_normalize_llm_label(k), -1)
            if idx >= 0:
                probs[idx] = max(probs[idx], _value_to_probability(v))

    if probs.sum() <= 0.0 and llm_idx >= 0:
        probs[llm_idx] = 1.0

    s = probs.sum()
    if s > 0.0:
        probs = probs / s

    pred_val = None
    if isinstance(conf_raw, dict):
        if llm_label_norm and llm_label_norm in conf_raw:
            pred_val = conf_raw[llm_label_norm]
        elif llm_label_norm and llm_label_norm.lower() in conf_raw:
            pred_val = conf_raw[llm_label_norm.lower()]
        elif conf_raw:
            pred_val = max(conf_raw.values())
    else:
        pred_val = conf_raw

    pred_prob = _value_to_probability(pred_val)
    if pred_prob <= 0.0 and llm_idx >= 0:
        pred_prob = float(probs[llm_idx]) if probs.sum() > 0.0 else 0.0

    return {
        "probs": probs,
        "idx": int(llm_idx),
        "label_raw": llm_label_raw,
        "conf_raw": conf_raw,
        "pred_prob": pred_prob,
    }


def _build_gnn_probs(gnn_res: Dict[str, Any], num_classes: int) -> Dict[str, Any]:
    probs = np.zeros(num_classes, dtype=np.float64)

    gnn_idx = gnn_res.get("gnn_predict")
    gnn_idx = int(gnn_idx) if gnn_idx is not None else -1

    gnn_probs = gnn_res.get("gnn_probs")
    if isinstance(gnn_probs, dict):
        for k, v in gnn_probs.items():
            try:
                idx = int(k)
            except (TypeError, ValueError):
                continue
            if 0 <= idx < num_classes:
                try:
                    probs[idx] = max(0.0, float(v))
                except (TypeError, ValueError):
                    continue

    if probs.sum() <= 0.0 and gnn_idx >= 0:
        probs[gnn_idx] = 1.0

    s = probs.sum()
    if s > 0.0:
        probs = probs / s

    return {"probs": probs, "idx": int(gnn_idx)}


def optimize_soft_energy(
    data,
    llm_preds: Dict[int, Dict[str, Any]],
    gnn_preds: Dict[int, Dict[str, Any]],
    inv_map: Dict[str, int],
    num_classes: int,
    cfg: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    eps = float(cfg.get("eps", 1e-12))
    lambda_clarity = float(cfg.get("lambda_clarity", 0.5))
    lambda_structure = float(cfg.get("lambda_structure", 1.0))
    lambda_fidelity = float(cfg.get("lambda_fidelity", 1.0))
    lambda_conflict = float(cfg.get("lambda_conflict", 1.5))
    lr = float(cfg.get("lr", 0.05))
    max_steps = int(cfg.get("max_steps", 80))
    tol = float(cfg.get("tol", 1e-5))
    conflict_boost = float(cfg.get("conflict_boost", 2.0))

    total_nodes = int(data.num_nodes)
    llm_probs = np.zeros((total_nodes, num_classes), dtype=np.float64)
    gnn_probs = np.zeros((total_nodes, num_classes), dtype=np.float64)
    llm_idx = np.full(total_nodes, -1, dtype=np.int64)
    gnn_idx = np.full(total_nodes, -1, dtype=np.int64)

    llm_label_raw_list = [None] * total_nodes
    llm_conf_raw_list = [None] * total_nodes
    llm_pred_prob = np.zeros(total_nodes, dtype=np.float64)

    for node_idx in range(total_nodes):
        llm_pack = _build_llm_probs(llm_preds.get(node_idx, {}), inv_map, num_classes)
        gnn_pack = _build_gnn_probs(gnn_preds.get(node_idx, {}), num_classes)

        llm_probs[node_idx] = llm_pack["probs"]
        gnn_probs[node_idx] = gnn_pack["probs"]
        llm_idx[node_idx] = llm_pack["idx"]
        gnn_idx[node_idx] = gnn_pack["idx"]
        llm_label_raw_list[node_idx] = llm_pack["label_raw"]
        llm_conf_raw_list[node_idx] = llm_pack["conf_raw"]
        llm_pred_prob[node_idx] = float(llm_pack["pred_prob"])

    llm_conf = llm_pred_prob
    gnn_conf = gnn_probs.max(axis=1)

    conflict_mask = (llm_idx >= 0) & (gnn_idx >= 0) & (llm_idx != gnn_idx)

    q = llm_conf / (llm_conf + gnn_conf + eps)
    z = np.where(conflict_mask, (llm_conf >= gnn_conf).astype(np.float64), q)

    w_clarity = 1.0 + conflict_boost * conflict_mask.astype(np.float64)
    w_fidelity = 1.0 + 0.5 * conflict_boost * conflict_mask.astype(np.float64)
    c_conflict = conflict_mask.astype(np.float64) * conflict_boost

    alpha = np.clip(q.copy(), 0.0, 1.0)

    edge_index = data.edge_index.cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    logger.info(
        "soft_energy cfg: "
        f"l_clarity={lambda_clarity}, l_structure={lambda_structure}, "
        f"l_fidelity={lambda_fidelity}, l_conflict={lambda_conflict}, "
        f"lr={lr}, max_steps={max_steps}, tol={tol}, conflict_boost={conflict_boost}"
    )

    rel_change = None
    for step in range(max_steps):
        p = alpha[:, None] * llm_probs + (1.0 - alpha[:, None]) * gnn_probs
        p = np.clip(p, eps, 1.0)

        grad_clarity = np.sum((np.log(p) + 1.0) * (gnn_probs - llm_probs), axis=1)
        grad_clarity = lambda_clarity * w_clarity * grad_clarity

        diff = alpha[src] - alpha[dst]
        grad_structure = np.bincount(src, weights=2.0 * diff, minlength=total_nodes)
        grad_structure += np.bincount(dst, weights=-2.0 * diff, minlength=total_nodes)
        grad_structure = lambda_structure * grad_structure

        grad_fidelity = lambda_fidelity * (2.0 * w_fidelity * (alpha - q))
        grad_conflict = lambda_conflict * (2.0 * c_conflict * (alpha - z))

        grad = grad_clarity + grad_structure + grad_fidelity + grad_conflict

        new_alpha = np.clip(alpha - lr * grad, 0.0, 1.0)

        rel_change = np.linalg.norm(new_alpha - alpha) / (np.linalg.norm(alpha) + eps)
        alpha = new_alpha

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(f"soft_energy step={step+1}/{max_steps} rel_change={rel_change:.6e}")

        if rel_change < tol:
            logger.info(f"soft_energy converged at step={step+1} rel_change={rel_change:.6e}")
            break

    p_final = alpha[:, None] * llm_probs + (1.0 - alpha[:, None]) * gnn_probs
    final_idx = p_final.argmax(axis=1).astype(np.int64)

    source = np.full(total_nodes, "mix", dtype=object)
    source[(final_idx == llm_idx) & (final_idx != gnn_idx)] = "llm"
    source[(final_idx == gnn_idx) & (final_idx != llm_idx)] = "gnn"
    source[(final_idx == llm_idx) & (final_idx == gnn_idx) & (final_idx >= 0)] = "both"

    return {
        "alpha": alpha,
        "final_idx": final_idx,
        "source": source,
        "llm_idx": llm_idx,
        "gnn_idx": gnn_idx,
        "llm_confidence_prob": llm_conf,
        "llm_label_raw": llm_label_raw_list,
        "llm_conf_raw": llm_conf_raw_list,
        "conflict_mask": conflict_mask,
        "diagnostics": {
            "steps_configured": max_steps,
            "final_rel_change": float(rel_change) if rel_change is not None else None,
            "mean_alpha": float(alpha.mean()),
            "conflict_ratio": float(conflict_mask.mean()),
        },
    }


def optimize_conflict_soft_energy(
    data,
    llm_preds: Dict[int, Dict[str, Any]],
    gnn_preds: Dict[int, Dict[str, Any]],
    inv_map: Dict[str, int],
    num_classes: int,
    cfg: Dict[str, Any],
    logger,
) -> Dict[str, Any]:
    eps = float(cfg.get("eps", 1e-12))
    lambda_clarity = float(cfg.get("lambda_clarity", 0.5))
    lambda_structure = float(cfg.get("lambda_structure", 0.0))
    lambda_fidelity = float(cfg.get("lambda_fidelity", 1.0))
    lambda_conflict = float(cfg.get("lambda_conflict", 2.0))
    lr = float(cfg.get("lr", 0.003))
    max_steps = int(cfg.get("max_steps", 150))
    tol = float(cfg.get("tol", 1e-6))
    conflict_boost = float(cfg.get("conflict_boost", 3.0))
    hard_threshold = float(cfg.get("hard_threshold", 0.99))
    route_mode = str(cfg.get("route_mode", "expert_pick")).strip().lower()
    conflict_min_conf_gap = float(cfg.get("conflict_min_conf_gap", 0.0))
    conflict_min_margin = float(cfg.get("conflict_min_margin", 0.0))
    conflict_min_margin_gap = float(cfg.get("conflict_min_margin_gap", 0.0))
    fallback_non_active_to_hard = bool(cfg.get("fallback_non_active_to_hard", True))
    llm_margin_weight = float(cfg.get("llm_margin_weight", 0.0))
    gnn_margin_weight = float(cfg.get("gnn_margin_weight", 0.0))
    llm_agree_weight = float(cfg.get("llm_agree_weight", 0.0))
    gnn_agree_weight = float(cfg.get("gnn_agree_weight", 0.0))
    score_temp = float(cfg.get("score_temp", 0.05))
    lambda_sparse = float(cfg.get("lambda_sparse", 0.0))
    lambda_balance = float(cfg.get("lambda_balance", 0.0))
    balance_target_raw = cfg.get("balance_target", "auto")

    total_nodes = int(data.num_nodes)
    llm_probs = np.zeros((total_nodes, num_classes), dtype=np.float64)
    gnn_probs = np.zeros((total_nodes, num_classes), dtype=np.float64)
    llm_idx = np.full(total_nodes, -1, dtype=np.int64)
    gnn_idx = np.full(total_nodes, -1, dtype=np.int64)

    llm_label_raw_list = [None] * total_nodes
    llm_conf_raw_list = [None] * total_nodes
    llm_pred_prob = np.zeros(total_nodes, dtype=np.float64)

    for node_idx in range(total_nodes):
        llm_pack = _build_llm_probs(llm_preds.get(node_idx, {}), inv_map, num_classes)
        gnn_pack = _build_gnn_probs(gnn_preds.get(node_idx, {}), num_classes)

        llm_probs[node_idx] = llm_pack["probs"]
        gnn_probs[node_idx] = gnn_pack["probs"]
        llm_idx[node_idx] = llm_pack["idx"]
        gnn_idx[node_idx] = gnn_pack["idx"]
        llm_label_raw_list[node_idx] = llm_pack["label_raw"]
        llm_conf_raw_list[node_idx] = llm_pack["conf_raw"]
        llm_pred_prob[node_idx] = float(llm_pack["pred_prob"])

    llm_conf = llm_pred_prob
    gnn_conf = gnn_probs.max(axis=1)
    conflict_mask = (llm_idx >= 0) & (gnn_idx >= 0) & (llm_idx != gnn_idx)

    llm_sorted = np.sort(llm_probs, axis=1)
    gnn_sorted = np.sort(gnn_probs, axis=1)
    llm_margin = llm_sorted[:, -1] - llm_sorted[:, -2]
    gnn_margin = gnn_sorted[:, -1] - gnn_sorted[:, -2]

    conf_gap = np.abs(llm_conf - gnn_conf)
    margin_gap = np.abs(llm_margin - gnn_margin)
    signal_gate = (
        (conf_gap >= conflict_min_conf_gap)
        | (np.maximum(llm_margin, gnn_margin) >= conflict_min_margin)
        | (margin_gap >= conflict_min_margin_gap)
    )
    active_conflict_mask = conflict_mask & signal_gate


    edge_index = data.edge_index.cpu().numpy()
    src = edge_index[0].astype(np.int64)
    dst = edge_index[1].astype(np.int64)

    # Unsupervised neighborhood agreement per expert.
    deg = np.bincount(src, minlength=total_nodes).astype(np.float64)
    llm_same = (llm_idx[src] >= 0) & (llm_idx[dst] >= 0) & (llm_idx[src] == llm_idx[dst])
    gnn_same = (gnn_idx[src] >= 0) & (gnn_idx[dst] >= 0) & (gnn_idx[src] == gnn_idx[dst])
    llm_agree = np.bincount(src, weights=llm_same.astype(np.float64), minlength=total_nodes)
    gnn_agree = np.bincount(src, weights=gnn_same.astype(np.float64), minlength=total_nodes)
    llm_agree = llm_agree / (deg + eps)
    gnn_agree = gnn_agree / (deg + eps)

    llm_score = llm_conf + llm_margin_weight * llm_margin + llm_agree_weight * llm_agree
    gnn_score = gnn_conf + gnn_margin_weight * gnn_margin + gnn_agree_weight * gnn_agree
    score_diff = llm_score - gnn_score
    q = 1.0 / (1.0 + np.exp(-score_diff / max(score_temp, eps)))
    z = (llm_score >= gnn_score).astype(np.float64)

    # Keep non-active nodes fixed to hard-threshold style decisions.
    fixed_alpha = np.where(llm_conf >= hard_threshold, 1.0, 0.0)
    fixed_alpha = np.where(llm_idx < 0, 0.0, fixed_alpha)
    fixed_alpha = np.where((gnn_idx < 0) & (llm_idx >= 0), 1.0, fixed_alpha)

    alpha = np.where(active_conflict_mask, q, fixed_alpha)
    alpha = np.clip(alpha, 0.0, 1.0)

    active_count = float(max(int(active_conflict_mask.sum()), 1))
    if isinstance(balance_target_raw, str) and balance_target_raw.lower() == "auto":
        balance_target = float(np.mean(q[active_conflict_mask])) if active_conflict_mask.any() else 0.5
    else:
        try:
            balance_target = float(balance_target_raw)
        except (TypeError, ValueError):
            balance_target = 0.5
    balance_target = max(0.0, min(1.0, balance_target))

    edge_conflict = active_conflict_mask[src] & active_conflict_mask[dst]

    c_conflict = active_conflict_mask.astype(np.float64) * conflict_boost

    logger.info(
        "conflict_soft_energy cfg: "
        f"l_clarity={lambda_clarity}, l_structure={lambda_structure}, "
        f"l_fidelity={lambda_fidelity}, l_conflict={lambda_conflict}, "
        f"lr={lr}, max_steps={max_steps}, tol={tol}, "
        f"conflict_boost={conflict_boost}, hard_threshold={hard_threshold}, "
        f"route_mode={route_mode}, min_conf_gap={conflict_min_conf_gap}, "
        f"min_margin={conflict_min_margin}, min_margin_gap={conflict_min_margin_gap}, "
        f"fallback_non_active_to_hard={fallback_non_active_to_hard}, "
        f"llm_margin_w={llm_margin_weight}, gnn_margin_w={gnn_margin_weight}, "
        f"llm_agree_w={llm_agree_weight}, gnn_agree_w={gnn_agree_weight}, score_temp={score_temp}, "
        f"lambda_sparse={lambda_sparse}, lambda_balance={lambda_balance}, balance_target={balance_target}"
    )

    rel_change = None
    for step in range(max_steps):
        p = alpha[:, None] * llm_probs + (1.0 - alpha[:, None]) * gnn_probs
        p = np.clip(p, eps, 1.0)

        grad_clarity = np.sum((np.log(p) + 1.0) * (gnn_probs - llm_probs), axis=1)
        grad_clarity = lambda_clarity * c_conflict * grad_clarity

        diff = alpha[src] - alpha[dst]
        diff = np.where(edge_conflict, diff, 0.0)
        grad_structure = np.bincount(src, weights=2.0 * diff, minlength=total_nodes)
        grad_structure += np.bincount(dst, weights=-2.0 * diff, minlength=total_nodes)
        grad_structure = lambda_structure * grad_structure

        grad_fidelity = lambda_fidelity * (2.0 * c_conflict * (alpha - q))
        grad_conflict = lambda_conflict * (2.0 * c_conflict * (alpha - z))
        grad_sparse = lambda_sparse * c_conflict * (1.0 - 2.0 * alpha)

        mean_alpha_active = float(np.sum(alpha * active_conflict_mask) / active_count)
        grad_balance_scalar = lambda_balance * 2.0 * (mean_alpha_active - balance_target) / active_count
        grad_balance = np.where(active_conflict_mask, grad_balance_scalar, 0.0)

        grad = grad_clarity + grad_structure + grad_fidelity + grad_conflict + grad_sparse + grad_balance
        grad = np.where(active_conflict_mask, grad, 0.0)

        new_alpha = alpha.copy()
        new_alpha[active_conflict_mask] = np.clip(
            alpha[active_conflict_mask] - lr * grad[active_conflict_mask], 0.0, 1.0
        )
        new_alpha[~active_conflict_mask] = fixed_alpha[~active_conflict_mask]

        rel_change = np.linalg.norm(new_alpha - alpha) / (np.linalg.norm(alpha) + eps)
        alpha = new_alpha

        if (step + 1) % 10 == 0 or step == 0:
            logger.info(
                f"conflict_soft_energy step={step+1}/{max_steps} rel_change={rel_change:.6e}"
            )

        if rel_change < tol:
            logger.info(
                f"conflict_soft_energy converged at step={step+1} rel_change={rel_change:.6e}"
            )
            break

    # Final decision: keep non-conflict as hard-threshold, conflict nodes by routing policy.
    p_final = alpha[:, None] * llm_probs + (1.0 - alpha[:, None]) * gnn_probs
    if route_mode == "fused_argmax":
        final_idx = p_final.argmax(axis=1).astype(np.int64)
        source = np.full(total_nodes, "mix", dtype=object)
        source[(final_idx == llm_idx) & (final_idx != gnn_idx)] = "llm"
        source[(final_idx == gnn_idx) & (final_idx != llm_idx)] = "gnn"
        source[(final_idx == llm_idx) & (final_idx == gnn_idx) & (final_idx >= 0)] = "both"
    else:
        choose_llm = alpha >= 0.5
        final_idx = np.where(choose_llm, llm_idx, gnn_idx)
        source = np.where(choose_llm, "llm", "gnn").astype(object)

    if fallback_non_active_to_hard:
        non_active_llm = (~active_conflict_mask) & (fixed_alpha >= 0.5)
        non_active_gnn = (~active_conflict_mask) & (fixed_alpha < 0.5)
        final_idx = np.where(non_active_llm, llm_idx, final_idx)
        final_idx = np.where(non_active_gnn, gnn_idx, final_idx)
        source = np.where(non_active_llm, "llm", source)
        source = np.where(non_active_gnn, "gnn", source)

    return {
        "alpha": alpha,
        "final_idx": final_idx.astype(np.int64),
        "source": source,
        "llm_idx": llm_idx,
        "gnn_idx": gnn_idx,
        "llm_confidence_prob": llm_conf,
        "llm_label_raw": llm_label_raw_list,
        "llm_conf_raw": llm_conf_raw_list,
        "conflict_mask": conflict_mask,
        "diagnostics": {
            "steps_configured": max_steps,
            "final_rel_change": float(rel_change) if rel_change is not None else None,
            "mean_alpha": float(alpha.mean()),
            "conflict_ratio": float(conflict_mask.mean()),
            "active_conflict_ratio": float(active_conflict_mask.mean()),
            "route_mode": route_mode,
            "hard_threshold": hard_threshold,
        },
    }
