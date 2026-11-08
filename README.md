# Skill Recommendation via Link Prediction

Predicting which skills the job market will demand **together**, from a skill
co-occurrence network built out of real job postings.

**Course:** Social Network Analysis (AAT). **Status:** complete (D1-D5).

Nodes are skills, an edge joins two skills that appear in the same posting, and the
edge weight counts the postings in which both appear. Twenty percent of the edges are
hidden, every remaining skill pair is scored with Common Neighbours, Jaccard,
Adamic-Adar, Preferential Attachment and Resource Allocation, and the scores are
evaluated with Precision@K, Recall@K, F1@K and ROC-AUC over 25 random splits plus a
temporal hold-out. The observed graph is then used for the syllabus analyses
(communities, homophily, bridges, structural holes, core-periphery, PageRank, HITS,
small-world), and the result is a filtered Top-20 list of skill pairs that are not yet
connected, plus a `recommend(skill, k)` function.

## Results at a glance

| Quantity | Value |
|---|---|
| Postings used | 27,846 of 33,246 (83.8 percent) |
| Skills in the network | 165 (157 in the giant component) |
| Edges | 883 at weight >= 50 and PMI >= 0.5 |
| Test edges per split | 173 hidden edges, 11,536 candidate pairs |
| Best single index | Resource Allocation, AUC 0.8837 |
| Best Precision@20 | Adamic-Adar, 0.142 (about 9.5x the 1.5 percent base rate) |
| Best at K = 10 | Common Neighbours, 0.152 |
| Weakest | Preferential Attachment, AUC 0.7787 |
| Random Forest | AUC 0.9189, 25 splits |
| Temporal hold-out | Common Neighbours P@20 0.45; Jaccard collapses to 0.00 |
| Communities | 7, modularity 0.528, NMI against curated domains 0.499 |
| Homophily | 75.2 percent of edges inside a community, baseline 19.1 percent |
| Small-world | sigma 6.06, clustering 0.5145 vs 0.0726 random |
| Top predicted pair | Python and Stakeholder Management (1.540, 22 shared neighbours) |

Full numbers: [`data/processed/results.json`](data/processed/results.json). The
report is [`docs/report.pdf`](docs/report.pdf), the deck is
[`docs/deck.pptx`](docs/deck.pptx).

## Repository layout

```
README.md
requirements.txt
data/
  raw/job_postings_hf.csv        # source postings (not committed, see Dataset)
  processed/                     # cleaned long table, node/edge tables, predictions, results.json
  README.md                      # dataset package: provenance, schema, cleaning log
src/
  pipeline.py                    # the whole pipeline, stages 1-13, runnable top to bottom
  skills_vocab.py                # curated vocabulary: 245 skills, 19 domains, aliases
notebooks/
  sna_skill_recommendation.ipynb # D1, executed, with markdown between every section
  stage1_data_check.ipynb        # early Stage 1 scaffold, kept for history only
tools/
  build_notebook.py              # regenerates the notebook from src/pipeline.py
  md_to_pdf.py                   # renders docs/report.md to docs/report.pdf
  make_deck.py                   # builds docs/deck.pptx and docs/deck_outline.md
docs/
  report.md, report.pdf          # D4
  deck.pptx, deck_outline.md     # D5
  cytoscape_guide.md             # D3 styling instructions
  resume_bullets.md
figures/                         # fig1-fig7, PNG (200 dpi) and SVG
graph/                           # GraphML, GEXF, node and edge CSVs for Cytoscape/Gephi
viz/top20_predicted_links.html   # interactive pyvis view
```

## Quickstart

```bash
python -m pip install -r requirements.txt
python src/pipeline.py                      # ~4 minutes end to end

python tools/build_notebook.py --execute    # rebuild and run the D1 notebook
python tools/md_to_pdf.py                   # rebuild the report PDF
python tools/make_deck.py                   # rebuild the deck and its outline
```

Colab: upload the repository, `cd` into it, and run the notebook top to bottom. The
pipeline resolves paths from the project root, so run it from there.

Configuration lives in one dictionary at the top of `src/pipeline.py`:
`min_postings=100`, `edge_threshold=50`, `pmi_min=0.5`, `test_frac=0.2`, `n_seeds=25`.

## Dataset

**Used here:** LinkedIn job postings in the `arshkon/linkedin-job-postings` schema,
downloaded as a single file from the Hugging Face mirror
[`xanderios/linkedin-job-postings`](https://huggingface.co/datasets/xanderios/linkedin-job-postings)
(`job_postings.csv`), 33,246 postings, 28 columns, 10 August to 4 November 2023.

**Also supported:** the Kaggle dataset in the brief,
[LinkedIn Jobs and Skills 2024](https://www.kaggle.com/datasets/asaniczka/1-3m-linkedin-jobs-and-skills-2024)
(`job_skills.csv`). Drop it into `data/raw/` and the pipeline loads it in
`skills_table` mode instead of extracting skills from descriptions. No stage after
loading changes.

Skills are extracted from free text with a curated vocabulary of 245 skills and 19
domains (`src/skills_vocab.py`), using greedy longest-match over token n-grams and a
checked alias table. There is no fuzzy matching: it merges `java` with `javascript`
and `c` with `c++`, which corrupts the graph invisibly.

Raw postings are not committed (size and licence); the download step above reproduces
them exactly. See [`data/README.md`](data/README.md) for schema, cleaning log and
caveats.

## Method summary

1. **Cleaning.** Deduplicate skills within a posting; keep skills in at least 100
   postings; keep postings with at least two recognised skills.
2. **Graph.** Project the posting-skill incidence onto skills; an edge needs support
   (weight >= 50) *and* selectivity (PMI >= 0.5). Keep the giant component. The
   threshold is chosen from a 15-point sensitivity grid, not by taste.
3. **Split.** Hide a random 20 percent of edges; repair connectivity (4 edges) and
   exclude repaired edges from the test set. Repeat over 25 seeds.
4. **Candidates and negatives.** Score all 11,536 non-edges (exact Precision@K), and
   report both 1:1 random negatives and distance-2 hard negatives.
5. **Indices.** CN, Jaccard, AA, PA and RA implemented by hand, cross-checked against
   NetworkX (maximum absolute difference 0).
6. **Models.** Logistic Regression and Random Forest on the five indices plus endpoint
   degrees, fitted on half of the held-out positives and scored on the full candidate
   set.
7. **Temporal validation.** Train before the median posting date, test on pairs that
   appear only afterwards, with the later window subsampled to equal size.
8. **SNA analysis.** Degree distribution and log-log fit, clustering and transitivity,
   Louvain communities and modularity, NMI against curated domains, homophily with a
   random-non-edge baseline, betweenness and constraint, k-core core-periphery,
   PageRank and HITS, small-world sigma against 20 random graphs.
9. **Recommendations.** Score unobserved pairs with the best method, filter (at least
   5 shared neighbours, no generic endpoint, minimum degree 5), cap each skill at two
   appearances, and publish the Top-20 plus `recommend(skill, k)`.

## Key findings

* **Neighbourhood information is the signal.** Adamic-Adar and Resource Allocation
  rank best overall (AUC 0.8819 and 0.8837); Common Neighbours is best at the very top
  (P@10 0.152) and the most robust in the temporal test (P@20 0.45).
* **Degree alone fails.** Preferential Attachment is the worst index (AUC 0.7787,
  P@20 0.046) because the graph is slightly disassortative (-0.126) and its hubs are
  generalist skills.
* **Jaccard is too aggressive.** Dividing by the union penalises hub pairs twice, and
  it is the only index that collapses under the temporal shift (P@20 0.00, AUC 0.6548).
* **Domains are not communities.** Seven communities at modularity 0.528, but NMI
  against the curated domains is only 0.499: the big software community holds
  programming, cloud, data and delivery skills together because postings bundle them.
* **The connectors are not the most technical skills.** Stakeholder Management leads
  betweenness (0.1288), PageRank (0.0342) and HITS authority (0.0467); Problem Solving
  has the lowest constraint in the graph (0.0515), the structural-hole signature.
* **The useful recommendations are cross-community.** Python and Stakeholder
  Management (22 shared neighbours), Agile and Data Analysis, C# and REST API,
  Kubernetes and REST API, Excel and Payroll.

## Deliverables

| ID | Deliverable | File |
|---|---|---|
| D1 | Source code and executed notebook | `src/pipeline.py`, `notebooks/sna_skill_recommendation.ipynb` |
| D2 | Dataset package (cleaned data, edge list, node list, README) | `data/processed/`, `data/README.md` |
| D3 | Visualisation artefacts and Cytoscape instructions | `figures/`, `graph/`, `viz/`, `docs/cytoscape_guide.md` |
| D4 | Recommendation and evaluation report | `docs/report.md`, `docs/report.pdf` |
| D5 | Presentation deck with speaker notes | `docs/deck.pptx`, `docs/deck_outline.md` |

## Limitations

Co-occurrence is not complementarity: the graph records demand bundles, not the causal
claim that learning one skill creates demand for another. The corpus is a three-month,
English-language LinkedIn sample of 33,246 postings, and extraction recall is bounded
by a 245-skill vocabulary. The giant component has 157 nodes, so the degree tail is
truncated and seven communities is as fine-grained as the partition gets. The temporal
test has 100 positives and is directional rather than precise.

## Visualisation

Open `viz/top20_predicted_links.html` in a browser for the interactive view, or import
`graph/skill_network.graphml` into Cytoscape and follow
[`docs/cytoscape_guide.md`](docs/cytoscape_guide.md) for the exact style mappings
(node size by degree, colour by community, predicted edges dashed red).

## Credits

Built with NetworkX, pandas, scikit-learn, matplotlib, pyvis, reportlab and
python-pptx. Postings from LinkedIn, distributed through the
`arshkon/linkedin-job-postings` collection and its Hugging Face mirror. Please check
the dataset licence before redistributing any of the raw text.
