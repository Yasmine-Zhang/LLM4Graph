# Cache-Only Student Comparison

`GNNTrainer` accepts `model_type: gcn` (default) or `model_type: mlp` in its
configuration. `SimpleMLP` uses linear layers, hidden-layer BatchNorm, ReLU and
dropout; its forward signature accepts but ignores `edge_index`. It shares the
trainer, optimizer and loss with GCN. This is a graph/no-graph baseline, not an
independent semantic annotator.

Run the comparison from the repository root in the `llm4graph` environment:

```powershell
python -m src.gnn.cache_experiment --cache output/backup.20260915/Arxiv.GPT5/llm_predections.json --data-root output/temp/ogb_verified/arxiv --output output/Arxiv.GPT5.student_comparison.v1 --device cuda:0
python -m src.gnn.cache_experiment --cache output/backup.20260915/Arxiv.GPT6/llm_predictions.json --data-root output/temp/ogb_verified/arxiv --output output/Arxiv.GPT6.student_comparison.v1 --device cuda:0
python output/temp/analyze_student_comparison.py --experiment output/Arxiv.GPT5.student_comparison.v1 --output output/temp/GPT5.student_analysis.v1
python output/temp/analyze_student_comparison.py --experiment output/Arxiv.GPT6.student_comparison.v1 --output output/temp/GPT6.student_analysis.v1
```

`--data-root` must point to an intact official Arxiv directory containing `raw`,
`mapping` and `split/time`. Existing local graph files were found truncated during
setup; do not use them without integrity validation. Each teacher is a separate
experiment. No API client is initialized and existing output directories are
never overwritten. The current entry point intentionally supports the audited
new-format caches (`status`, `node_id`, `paper_id`, `model`, `protocol_id`,
`llm_predict`, `self_reported_confidence`), not legacy confidence encodings.

Defaults: 3 layers, hidden size 256, dropout 0.5, Adam lr 0.01, weight decay 0,
500 fixed epochs, seed 42, self-reported confidence >= 0.9. Use `--selection all`
for an all-pseudolabel baseline, or `--seeds 42 43 44` for repeated runs. Missing
confidence is excluded from threshold selection, not converted from logprob.
The 0.9 self-reported threshold is not calibrated or equivalent to the old Azure
token-logprob threshold. This is an exploratory configuration, not a tuned claim.

The training command loads only official 128-dimensional features, edges,
category names and node-to-paper IDs. It never loads true labels or official
splits. Nonanchor targets are -1; only anchor targets enter the loss. Graph edges
are symmetrized and coalesced as in the existing pipeline. Both models use
transductive BatchNorm across all nodes. Seeds are reset before each model;
this does not guarantee bitwise CUDA reproducibility or identical initialization
across different model classes. CUDA is required unless `--device cpu` is explicit.
The PyPI/Tsinghua Windows torch wheel installs a CPU build. CUDA requires the
official CUDA wheel; use `--device cpu` explicitly when that download is unavailable.
`--adjacency-format csr` uses PyTorch sparse adjacency for the same GCN operator;
the focused tests compare its outputs and gradients against the edge-list path.

For a shorter full-graph feasibility pilot, keep all other settings and use
`--epochs 50 --device cpu --adjacency-format csr` with an output directory ending
in `student_comparison.pilot50`. The pilot is not a completed 500-epoch baseline.
Do not compare different training budgets as though they were matched, or select
an epoch using the correction-space evaluation. The first pilot uses only seed 42.

Artifacts include input/source hashes, teacher identity, class mapping, anchor
mask/counts, loss history, final weights, logits, predictions and completion
markers. No checkpoint is selected using real validation labels. Unlike the
legacy `main.py` path, this command is safe for fixed-epoch label-free comparison;
choosing `mlp` in the legacy path does not remove its true-label model selection.

The separate analysis command reads truth only after training completes. It
reports all nodes, anchors, nonanchors and official split intersections. Official
test nodes can themselves be pseudolabel anchors; test accuracy is therefore not
automatically nonanchor accuracy. It also reports confidence/degree/class strata
and GCN-versus-MLP comparisons. An oracle union measures available correction
space, not an implementable router or evidence of achievable improvement.

Run focused and end-to-end tests:

```powershell
python output/temp/test_student_comparison.py -v
```