# Resume bullets

Two versions of the same project. Use one, not both, and adjust the numbers only if
you rerun the pipeline and get different ones.

## Version A - data science / machine learning emphasis

**Skill Recommendation via Link Prediction** | Python, NetworkX, scikit-learn, pandas
*Social Network Analysis course project*

* Built a 157-node, 883-edge skill co-occurrence network from 27,846 LinkedIn job
  postings, with a 245-skill curated vocabulary, alias normalisation and a
  support-plus-PMI edge rule chosen from a 15-point sensitivity grid.
* Implemented Common Neighbours, Jaccard, Adamic-Adar, Preferential Attachment and
  Resource Allocation from their formulas and verified them against NetworkX (zero
  deviation), then benchmarked them over 25 random 80/20 splits with Precision@K,
  Recall, F1 and ROC-AUC.
* Showed that Adamic-Adar and Resource Allocation rank best (AUC 0.884) while
  Preferential Attachment fails (0.779), and that a Random Forest on the five indices
  reaches 0.919; validated the ranking on a forward-in-time hold-out rather than only
  on randomly hidden edges.
* Turned the benchmark into a recommender that returns the Top-20 unobserved skill
  pairs and a `recommend(skill, k)` function, with filters for generic skills and a
  per-skill diversity cap.

## Version B - analyst / product emphasis

**Skill Recommendation via Link Prediction** | Python, NetworkX, matplotlib
*Social Network Analysis course project*

* Modelled the labour market as a skill network: nodes are skills, edges are
  co-demand in the same job posting, weights count postings, built from 27,846
  postings and 165 skills.
* Quantified the structure: 7 communities (modularity 0.528), small-world coefficient
  6.06, and 75 percent of edges inside a community against a 19 percent baseline.
* Produced a Top-20 list of skill pairs the market is likely to demand together but
  has not yet connected, led by Python and Stakeholder Management, and interpreted each
  cluster of recommendations in plain language.
* Delivered the full package - executed notebook, cleaned dataset, Cytoscape-ready
  graph, 15-page report and 15-slide deck - so the analysis is reproducible from two
  commands.

## One-line summary (portfolio or LinkedIn)

Built a link-prediction model on a 27,846-posting skill co-occurrence network that
identifies which skill pairs the job market is likely to demand together next, and
packaged it as a recommender, a report and a deck.
