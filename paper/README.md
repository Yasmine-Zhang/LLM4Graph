# Manuscript and Reproduction

## English Conference-Style Review Draft (27 September 2026)

The [15-page English PDF](neurips_rethink.pdf), [main source](neurips_rethink.tex) and [supplementary source](neurips_rethink_appendix.tex) present the current three-question argument, full H/I frozen comparison families, derivations, protocol, evidence boundaries and resource accounting. Pages 1--7 contain the body and references; the appendix starts on page 8. The named authors are Yapu Zhang (Beijing University of Technology) and Xianliang Yang (Microsoft, corresponding author). This is shorter than an eight-page body plus a roughly 20-page supplement: no independent new-cohort results, matched end-to-end costs or human judgments exist to justify that length. The [Chinese working draft](../paper.md) retains the fuller research history and diagnostics.

The manuscript uses the official [NeurIPS 2025 style](neurips_2025.sty) (from the [official style archive](https://media.neurips.cc/Conferences/NeurIPS2025/Styles.zip)) in named-author preprint layout. It is an unsubmitted working draft, not a 2026-compliant template or a finished NeurIPS/ICLR/ICML submission. The default preprint notice is replaced with an explicit unsubmitted-draft notice. The [read-only evidence check](../paper_evidence_check.pl) validates local claims against frozen files, not independent reproduction.

Compile from this directory with Tectonic and the included style file:

```bash
tectonic -r 2 neurips_rethink.tex
```

The committed PDF is a reading copy; do not overwrite it when building locally. The [evidence checker](../paper_evidence_check.pl) reads the Chinese working draft and the frozen experiment records under `output/analysis`. Run `perl -w paper_evidence_check.pl` from the repository root only where the separate, authorized local analysis archive is available. Before any submission, select the actual target venue/year, use its current official instructions and checklist, and independently review evidence, permissions and author information.

The following sections are local archival notes about earlier drafts and builders, not dependencies of the current English manuscript. Links under `output/` are available only with the local analysis archive; they are not included in this repository.

## Current Reading Copy (26 September 2026)

The [Chinese PDF of the current working draft](../output/analysis/paper_current_pdf_20260926_v1/main.pdf) is generated from [paper.md](../paper.md); its [manifest](../output/analysis/paper_current_pdf_20260926_v1/manifest.json) records the exact Markdown and PDF hashes. It is a searchable reading copy, not an independently confirmed or submission-ready method paper. H/I have been exposed; the [candidate preflight and exposed-H microbenchmarks](../output/analysis/70_CANDIDATE_COHORT_AND_EQUAL_FACT_RESOURCE.md) do not complete independent confirmation or matched end-to-end resource comparison. Human semantic review remains unperformed and is assigned to the researcher. The separate English LaTeX/PDF v6 below is an older snapshot and does **not** represent this draft.

The local [export script](export_current.py) uses `markdown-it-py`, `mdit-py-plugins`, KaTeX, a Chinese font and WeasyPrint 65.1. It requires explicit `--out`, `--node`, `--katex` and `--font` paths and refuses to overwrite an existing output directory. The PDF manifest and `paper_evidence_check.pl` verify different things: source identity and layout versus frozen experimental evidence. The PDF does not replace the evidence archive or permission review.

Latest evidence is in [report 49](../output/analysis/49_QUERY_FUSION_AND_CALIBRATED_REPLACEMENT.md): query-level graph fusion has four positive adjusted comparisons on 384 fresh TRAIN queries, but does not beat the strongest simple rule. [../paper.md](../paper.md) contains the updated result and limits. These results are not yet incorporated in the older PDF snapshot below.

The latest self-contained research manuscript is [../paper.md](../paper.md), including candidate-budget and local typed-decision experiments and a fresh 256-query transfer failure. Hosted Jev is prepared but not evaluated. The LaTeX/PDF below remains an older immutable snapshot.

This directory contains the current English LaTeX manuscript, its bibliography, and a data-driven builder. The paper is a research draft with no assigned authors and has not been submitted or uploaded. It does not claim a stable new performance-leading method.

Status on 2026-09-22: LaTeX/PDF v6 remains a round-12 snapshot. The current [extended draft](../paper_draft.md) and [report 46](../output/analysis/46_LINK_WEIGHTING_AND_INDEPENDENT_TRANSFER.md) add a 1,036-query independent confirmation and three TRAIN-only interventions; these are not yet typeset in v6. All validation queries are now exposed, while official test/human data remain unused. Do not use v6's earlier data-availability statements as the current research status.

## Reading

- [Current 17-page PDF](../output/analysis/paper_build_20260921_v6/main.pdf) and [manuscript source bundle](../output/analysis/paper_build_20260921_v6/manuscript_source.zip).
- [Build and evidence audit](../output/analysis/paper_build_20260921_v6/evidence_audit.json) and [fresh standalone rebuild](../output/analysis/paper_bundle_rebuild_20260921_v6/validation.json).
- [main.tex](main.tex): compact manuscript, proofs, experiments, limitations, and appendices.
- [references.bib](references.bib): bibliography with versioned primary sources.
- [../paper_draft.md](../paper_draft.md): longer editable English discussion; the LaTeX manuscript is the typeset reading version.
- [../paper_design.md](../paper_design.md): current Chinese research design and evidence map.

Generated build directories are immutable snapshots. The final build path is recorded in the latest research handoff, [../output/analysis/06_NEXT_SESSION_CONTEXT.md](../output/analysis/06_NEXT_SESSION_CONTEXT.md). Earlier builds are preserved, including compiler warnings; do not overwrite them.

## Local Build

From the repository root, using the existing ML interpreter:

```bash
/home/xianliang/miniconda3/envs/llm4graph/bin/python paper/build.py --out output/analysis/paper_build_NEW_VERSION
```

The output directory must not exist. The builder reads the already validated experiment JSONs, generates seven LaTeX result tables, copies two project figures, runs Tectonic 0.15.0, and checks the final PDF, references, missing glyphs, and overfull lines. A separate data-role table is in the source. Tectonic resides under `output/analysis/paper_tools_20260921`; its first execution may download TeX resources. It does not install into the system or change the editor interpreter. A second full engine run with fixed reruns stabilizes references and explicitly writes a new PDF; the TeX-only mode is insufficient for this check. The compiler output and source/data hashes are retained.

`--tables-only` checks and stages the inputs without compiling. Python 3.10+ and `pypdf` are required for the full builder. Table generation otherwise uses the standard library. The research runtime separately uses NumPy, scikit-learn, PyTorch/Transformers, and an isolated LightGBM 4.6.0 directory; the paper builder does not retrain any model.

Add `--package` to create the whitelisted manuscript source ZIP after a successful build. Current sources match v6, which adds post hoc structural partitions and exact factorization without replacing historical results. Neither subgroup descriptions nor synthetic scaling tests establish new method superiority. When testing the source bundle, exclude its existing `main.pdf` before compiling and use the full engine; `--pass tex` only writes an intermediate file and cannot validate a fresh PDF build. The successful standalone rebuild above follows this rule.

## Read-Only Evidence Audit

```bash
/home/xianliang/miniconda3/envs/llm4graph/bin/python paper/audit.py --build output/analysis/paper_build_VERSION
```

The audit checks the current manuscript sources against that build, regenerates its exact tables from verified JSON, verifies original confirmation source/model hashes, checks references and PDF text, and exhaustively tests finite matched-mask and error-decomposition cases. It makes no model/API calls. An optional `--out path.json` saves the report exclusively; omit it to rerun the audit without writes. A source edit correctly makes an older build fail the current-source check: create a new version rather than changing old manifests.

Full experiment validators write exclusive artifacts. Do not blindly rerun their write stages in completed directories. Their snapshots, results, and validation manifests are linked from the experiment reports; offline audit is the safe repeated check.

## Evidence Roles

The initial development cohorts contain 640 distinct official TRAIN queries. Original learned scorers fit A/B only, 256 queries. A later, separately frozen 512-query TRAIN acquisition expands some fits to A/B/F, 768 queries; C/D/E remain 384 repeatedly exposed development queries. Neither repeated arms, expanded candidate pairs, nor copied list views are independent queries.

The original 512-query validation evaluation and six contrasts remain unchanged. A later link-weighting method passed old-TRAIN development triage and was frozen before one-shot evaluation on the remaining 1,036 validation queries. Its eight primary strong-baseline comparisons did not establish superiority. All validation is now exposed; official test/human data remain unused. Later TRAIN-only representation interventions also fail their fixed gates. Historical failures are not rewritten, and the two validation cohorts are not pooled to manufacture significance.

## Distribution Boundary

A manuscript-only source bundle may contain only `main.tex`, `references.bib`, `main.bbl`, `tables.tex`, `table_data.json`, the two figure PDFs, the manuscript PDF, a build manifest, and a short build instruction. It must not recursively include `output/`, API responses, raw datasets, environment directories, or credential files. The local builder depends on the evidence tree; a standalone source bundle uses the already generated tables and bibliography and can be compiled with a normal LaTeX installation or Tectonic.

This is not a claim that a public end-to-end experiment package is complete. Raw dataset, model, service-output redistribution and source-code licensing require review. No license is granted on behalf of the user or third parties. Public upload additionally requires author and content approval. Secret scans do not replace that permission review.