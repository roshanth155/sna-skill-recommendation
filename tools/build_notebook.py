"""Build notebooks/sna_skill_recommendation.ipynb from src/pipeline.py.

The notebook is generated, never hand-edited, so the code in the notebook and the
code in src/pipeline.py can never drift apart. Each numbered section of the script
becomes one markdown cell (what this does and why) followed by one code cell.

Usage:  python tools/build_notebook.py [--execute]
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src" / "pipeline.py"
DST = ROOT / "notebooks" / "sna_skill_recommendation.ipynb"

TITLE = """# Skill Recommendation via Link Prediction

**Social Network Analysis course project (AAT)**

Nodes are skills, an edge joins two skills that appear in the same job posting, and the
edge weight counts how many postings contain both. The notebook hides 20% of the edges,
scores every remaining candidate pair with Common Neighbours, Jaccard, Adamic-Adar,
Preferential Attachment and Resource Allocation, evaluates them with Precision@K,
Recall, F1 and ROC-AUC over 25 random splits, then uses the surviving network for the
syllabus analyses (communities, homophily, bridges, core-periphery, PageRank, HITS,
small-world) and produces the Top-20 recommended skill pairs.

**How to run.** From the project root (so that `src/skills_vocab.py` is importable):
choose *Runtime -> Run all*. On Colab, upload the repository, then run:
`import os; os.chdir("sna-skill-recommendation")` (or wherever the repo landed).

**Fixed random seed** everywhere (`SEED = 42`, plus seeds `0..24` for the repeated
splits), so every number in the report is reproducible.

The cell below is the pipeline source: configuration, then one section per stage.
"""

NOTES = {
    0: """## 0. Setup

Imports, paths and the single configuration dictionary that controls every stage.
Keeping the thresholds here means the report can quote one place where they were set:
`min_postings=100` (rarity filter), `edge_threshold=50` and `pmi_min=0.5` (edge rule),
`test_frac=0.2` (hold-out share), `n_seeds=25` (repeated splits).""",
    1: """## 1. Load the postings

The pipeline accepts two input layouts and picks whichever file it finds:

* a **skills table** (`job_skills.csv`, one row per posting-skill pair, with a
  comma-separated cell) — the Kaggle layout named in the brief;
* **postings with a free-text description** — what this run uses, because the
  `skills_desc` field is only filled in about 1% of the rows.

Nothing is hard-coded to a guessed schema: the posting-id and description columns are
detected from the file that is actually present.""",
    2: """## 2. Skills and cleaning

Skills are read out of each description with the curated vocabulary in
`src/skills_vocab.py` (245 skills, 19 domains, alias phrases). Extraction is greedy
longest-match over token n-grams, so `machine learning` is consumed once and `sql`
can never be found inside `mysql`.

Two cleaning rules matter for correctness:

* **deduplicate within a posting** — a posting repeating `python` once, twice or three
  times must count once, otherwise co-occurrence weights are inflated;
* **rarity filter** — a skill must appear in at least 100 postings. Below that, an
  edge is supported by a handful of postings and Jaccard/Adamic-Adar are unstable.""",
    3: """## 3. The weighted co-occurrence graph

Every posting with *k* skills contributes one co-occurrence to each of its *k(k-1)/2*
pairs, so the edge weight is literally "in how many postings did these two appear
together".

**Why two conditions for an edge.** A raw count threshold alone keeps pairs that meet
by chance: Excel and Leadership are both in thousands of postings, so they clear any
count threshold. Pointwise mutual information asks whether the pair appears *more often
than the two skills' popularity predicts* (`log(observed / expected)`). The edge rule is
therefore **weight >= 50 and PMI >= 0.5**: enough support, and selective. The
sensitivity table shows the trade-off (edges, components, density, transitivity) across
a small grid, and the giant component is kept, as standard practice.""",
    4: """## 4. Train/test split and negative sampling

The main split hides a **random 20% of edges**. Because a random removal can cut a leaf
skill off the graph, edges that would disconnect the training graph are put back and
removed from the test set — they are never scored, so nothing leaks (`repaired` counts
them).

**Candidates.** Every pair that is not a training edge is scored (~11,500 pairs). On a
graph this size that is cheap, and it makes Precision@K an exact measurement rather than
an estimate from a sample. **Negatives** are then drawn two ways: uniform random
non-edges (the benchmark setting) and **distance-2 pairs** — skills that share a
neighbour but are not connected, which is the realistic "should these be linked?"
question and much harder.""",
    5: """## 5. Link-prediction scores

All five indices are implemented by hand from their formulas (a viva requirement) and
then cross-checked against NetworkX on 200 pairs; the assertion fails loudly if they
disagree.

* **CN** — number of shared neighbours.
* **Jaccard** — shared neighbours divided by the union of neighbourhoods; penalises hubs.
* **Adamic-Adar** — shared neighbours weighted by `1/log(degree)`, so rare connectors
  count more.
* **Preferential Attachment** — product of degrees; uses no neighbourhood at all.
* **Resource Allocation** — `1/degree`, a sharper Adamic-Adar (optional extra).""",
    6: """## 6. Evaluation: Precision@K, Recall, F1, ROC-AUC over 25 seeds

One split gives one noisy number, so the whole experiment is repeated over 25 random
splits and reported as **mean ± standard deviation**.

* **Precision@K** — of the top *K* ranked pairs, how many are truly held-out edges.
* **Recall@K** — of all held-out edges, how many appear in the top *K*.
* **F1@K** — harmonic mean of the two at that cut-off.
* **ROC-AUC** — ranking quality over the whole candidate set (threshold-free).

Two additional views are reported because they answer different questions: ranking
against **1:1 random negatives** (the usual benchmark, easy) and against **distance-2
hard negatives** (realistic, much harder).""",
    7: """## 7. Supervised comparison

Logistic Regression and Random Forest take the five heuristic scores plus the two
endpoint degrees as features.

**Leakage control:** features are computed from the training graph only, and the
held-out positives are split in half — one half fits the classifier, the other half is
the test set. The classifier is scored on the *same full candidate set* as the
heuristics, so the comparison is like-for-like rather than an easier 1:1 subset.""",
    8: """## 8. Temporal validation

The strongest guard against leakage: train on postings listed before the median date and
test on pairs that appear **only** in the later window. The later window is subsampled to
the same number of postings as the earlier one, so both graphs apply the same evidence
standard — otherwise the newer graph would look richer simply because more postings were
observed. This is a genuinely forward-looking test rather than a randomly hidden edge.""",
    9: """## 9. SNA analysis tied to the syllabus

* degree distribution and a log-log fit (is it heavy-tailed? this is why PA behaves as it does);
* clustering coefficient and transitivity (the triadic-closure basis of CN/Jaccard/AA);
* Louvain communities, modularity, and how far they align with the curated domains (NMI);
* homophily: share of observed vs predicted links that stay inside a community;
* betweenness and Burt's constraint (bridges and structural holes);
* k-core decomposition (core-periphery);
* PageRank and HITS (hubs/authorities);
* small-world check: clustering and average path length against a random graph of the
  same size and edge count, giving the small-world coefficient sigma.""",
    10: """## 10. Recommendations

The final list scores **every pair that is not an edge in the observed graph** — those are
the "not yet demanded together, but likely" candidates — with the best method from the
evaluation, then applies filters that a human reviewer would apply anyway:

* `cn >= 5` — one shared neighbour is a coincidence, not demand;
* **no generic skill** on either side (Communication, Leadership, ...): "Git + Leadership"
  is not something you can learn next;
* minimum degree 5;
* **diversity cap**: any skill appears at most twice in the Top-20, otherwise the
  strongest hub fills the whole list.

`recommend(skill, k)` applies the same logic to one skill, which is the "what should I
learn next given skill X" product.""",
    11: """## 11. Exports for Cytoscape / Gephi

`graph/skill_network.graphml` and `.gexf` carry node attributes (domain, degree community,
betweenness, constraint, PageRank, HITS, k-core) and edge attributes (weight, PMI,
predicted flag, score), so the styling instructions in `docs/cytoscape_guide.md` work
without any manual data editing. The Top-20 predicted links are included as edges with
`predicted=1` and `weight=0` so they can be styled as dashed.""",
    12: """## 12. Figures

Seven figures are written as both PNG (200 dpi, for the report and slides) and SVG
(vector, for print): the community map, degree distribution, edge-weight distribution
with the chosen threshold, ROC curves, metric comparison with error bars, the Top-20
predicted pairs, and the k-core profile. A pyvis HTML file gives an interactive view of
the predicted pairs in their neighbourhood.""",
    13: """## 13. Save results

Every number used in the report and the slide deck is written to
`data/processed/results.json`; the report generator reads that file rather than
re-typing values, so the prose can never drift from the experiment.""",
}


def main() -> None:
    lines = SRC.read_text(encoding="utf-8").splitlines()
    banner = re.compile(r"^#\s*-+\s*(\d+)\.\s*(.+)$")

    cells = [("markdown", TITLE)]
    current: list[str] | None = None
    section_no = None
    for line in lines:
        m = banner.match(line)
        if m:
            if current is not None:
                cells.append(("code", "\n".join(current).strip("\n")))
            section_no = int(m.group(1))
            note = NOTES.get(section_no)
            if note:
                cells.append(("markdown", note))
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        cells.append(("code", "\n".join(current).strip("\n")))

    nb_cells = []
    for kind, text in cells:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {},
                             "source": text.splitlines(True)})
        else:
            if not text.strip():
                continue
            nb_cells.append({"cell_type": "code", "execution_count": None,
                             "metadata": {}, "outputs": [],
                             "source": text.splitlines(True)})

    nb = {
        "cells": nb_cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    DST.parent.mkdir(parents=True, exist_ok=True)
    DST.write_text(json.dumps(nb, indent=1, ensure_ascii=False), encoding="utf-8")
    n_md = sum(1 for c in nb_cells if c["cell_type"] == "markdown")
    print(f"wrote {DST.relative_to(ROOT)}: {len(nb_cells)} cells ({n_md} markdown, "
          f"{len(nb_cells)-n_md} code)")

    if "--execute" in sys.argv:
        import nbclient
        import nbformat

        notebook = nbformat.read(DST, as_version=4)
        client = nbclient.NotebookClient(notebook, timeout=1800, kernel_name="python3",
                                         resources={"metadata": {"path": str(ROOT)}})
        client.execute()
        nbformat.write(notebook, DST)
        print("executed:", DST.relative_to(ROOT))


if __name__ == "__main__":
    main()
