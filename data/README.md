# Dataset package (D2)

## Source

* **Dataset:** LinkedIn job postings, in the schema of the `arshkon/linkedin-job-postings`
  collection.
* **File used:** `job_postings.csv`, downloaded from the Hugging Face mirror
  `xanderios/linkedin-job-postings` and stored as `data/raw/job_postings_hf.csv`.
* **Size:** 33,246 postings, 28 columns.
* **Date range in the file:** 2023-08-10 to 2023-11-04 (`listed_time`, epoch milliseconds).
* **Licence / terms:** as stated on the dataset page. Check it and quote it in the
  report before redistributing any raw text.
* **Also supported (not used in this run):** the Kaggle dataset
  `asaniczka/1-3m-linkedin-jobs-and-skills-2024`, whose `job_skills.csv` the pipeline
  reads in `skills_table` mode. Same schema downstream, no code change.

The postings carry a free-text `description`; extracted skills are not supplied as a
column. That is why cleaning and extraction are a substantive part of the project.

## Raw files

| File | Rows | Columns used | Use |
|---|---|---|---|
| `data/raw/job_postings_hf.csv` | 33,246 | `job_id`, `title`, `description`, `listed_time`, `formatted_experience_level`, `location` | skill extraction, dates for the temporal split |

Raw files are **not** committed (size and licence). Download the file above into
`data/raw/` to reproduce.

## Cleaning pipeline (`src/pipeline.py`, with `src/skills_vocab.py`)

1. **Column detection** - the posting id and description columns are found by name, so
   a differently named file still loads.
2. **Skill extraction** - greedy longest-match over token n-grams against a curated
   vocabulary of 245 skills in 19 domains. `machine learning` is consumed as one skill
   before `learning` can match, and `sql` can never be found inside `mysql`.
3. **Normalisation** - lowercase, unify unicode dashes, strip quotes and stray
   separators, collapse whitespace, strip trailing punctuation. `+`, `#`, `.`, `/` and
   `-` are preserved so `c++`, `c#`, `.net`, `node.js`, `ci/cd` and `scikit-learn`
   survive.
4. **Alias table** - checked synonyms only (`js` to `javascript`, `postgres` to
   `postgresql`, `k8s` to `kubernetes`). No fuzzy matching: it merges `java` with
   `javascript` and `c` with `c++`.
5. **Deduplicate within a posting** - a posting listing Python three times counts it
   once, otherwise co-occurrence weights are inflated.
6. **Rarity filter** - keep skills in at least `min_postings = 100` postings. Below
   that, Jaccard and Adamic-Adar divide by tiny neighbourhoods and an edge is a handful
   of postings rather than a pattern.
7. **Graph** - project the posting-skill incidence onto skills; an edge needs
   `weight >= 50` and `PMI >= 0.5`; keep the giant component.

## Cleaning log (real run)

| Step | Result |
|---|---|
| Postings loaded | 33,246 |
| Skill mentions extracted | 155,027 |
| Postings with at least 2 recognised skills | 27,846 (83.8 percent of the file) |
| Skills in the curated vocabulary | 245 |
| Skills after the rarity filter | 165 |
| Postings after the rarity filter | 27,846 |
| Mean skills per posting | 5.57 (median 5, max 30) |
| Distinct co-occurring pairs | 10,764 (max weight 4,788) |
| Graph at t = 50, PMI >= 0.5 | 165 nodes, 885 edges, 7 components |
| Giant component | 157 nodes, 883 edges, density 0.0721 |
| Isolated skills dropped | Android, Communication, HVAC, PHP, Preventive Maintenance, VPN, Welding, iOS |

## Output schema

`data/processed/skills_long.csv.gz` - one row per (posting, canonical skill),
deduplicated. Every later stage reads only this file.

| Column | Type | Meaning |
|---|---|---|
| `job_id` | string | posting identifier |
| `skill` | string | canonical skill name |

`data/processed/nodes.csv` - one row per node of the giant component:
`skill, domain, degree, weighted_degree, community, betweenness, constraint, pagerank,
hits_hub, hits_authority, kcore, generic`.

`data/processed/edges.csv` - `source, target, weight, predicted, score`. Observed edges
have `predicted = 0`; the Top-20 predicted pairs are appended with `predicted = 1` and
`weight = 0`, so they can be styled separately without contaminating weighted metrics.

`data/processed/predictions_top20.csv`, `predictions_top200.csv` - the ranked
recommendations with shared neighbours and domains.

`data/processed/metrics_by_method.csv`, `metrics_ml.csv`,
`metrics_sampled_negatives.csv` - the evaluation tables quoted in the report.

`data/processed/results.json` - every number used in the report and the deck. All
figures quoted in the write-up come from this file, not from memory.

`graph/skill_network.graphml`, `graph/skill_network.gexf` - the attributed graph for
Cytoscape and Gephi.

## Known caveats to state in the report

* **Coverage bias:** only postings with at least two recognised skills enter the
  network (83.8 percent of the file). Postings whose skills the vocabulary missed are
  invisible.
* **Time skew:** the file covers 10 August to 4 November 2023, roughly a quarter. The
  temporal split compares two windows of equal posting count but not equal calendar
  length.
* **Co-occurrence is not causation or complementarity:** two skills can co-occur
  because a job title bundles them. The main limitation of the model.
* **Vocabulary noise and ambiguity:** `Quality Assurance` covers both software QA and
  regulated manufacturing, and the shared neighbours of `Quality Assurance` and
  `Compliance` (GMP, HIPAA) show the manufacturing sense dominating. Stated in the
  report as a limitation, not a finding.
* **Truncated tail:** 165 vocabulary skills and 157 in the giant component, so the
  degree distribution is truncated at degree 53 and should not be fitted as a power law.

## Reproduce

```bash
python -m pip install -r requirements.txt
python src/pipeline.py
```

Writes `data/processed/`, `figures/`, `graph/` and `viz/`. Environment knobs are the
`CONFIG` dictionary at the top of `src/pipeline.py`; all randomness is seeded.
