# Deck outline (D5)

Generated from the same slide specification as `docs/deck.pptx`, with speaker notes.

## Slide 1. Which skills does the market demand together?
*Link prediction on a skill co-occurrence network built from 27,846 job postings*

**Speaker notes.** Open with the question, not the method: choosing a skill is a bet on a bundle. One sentence on the data (LinkedIn postings, Aug-Nov 2023) and one on the output (Top-20 skill pairs plus a learn-next function). Keep this slide under 30 seconds.

## Slide 2. A job posting is a bundle, not a list
- Python, SQL and AWS appear in one advert because one role needs all three
  - So the value of a skill depends on what it is used with
- Observed bundles give a network: skills are nodes, co-demand is an edge
- The interesting structure is what is missing, not what is popular
  - Link prediction: which unobserved pairs are likely to be demanded together?
- Output: Top-20 pairs the market will likely demand together, and learn-next(skill)

**Speaker notes.** Motivate the shift from single-skill popularity to pairs. The last bullet is the product; everything after this slide is the evidence behind it. Mention that the same machinery answers a second question: what to learn next given skill X.

## Slide 3. From postings to a skill network
- Bipartite incidence: postings x skills, 27,846 x 165 after cleaning
- Project onto skills: edge weight = postings mentioning both skills
- Undirected because co-occurrence is symmetric; direction was never in the data
- Two edge conditions: support (weight >= 50) and selectivity (PMI >= 0.5)
- Keep the giant component: 157 nodes, 883 edges, density 0.072
- Eight isolated skills reported, not silently dropped

**Speaker notes.** Explain the projection carefully; this is the modelling decision examiners probe. Undirected is not a simplification: a posting gives identical evidence both ways. The two-condition edge rule is the next slide's topic.

## Slide 4. 33,246 postings, three months, one vocabulary
- Source: LinkedIn postings (arshkon schema), Hugging Face mirror, Aug 10 - Nov 4 2023
  - Kaggle 1.3M skills table also supported: pipeline has a skills_table load path
- 245 curated skills, 19 domains, alias table, no fuzzy matching
  - Fuzzy matching would merge java with javascript and c with c++
- 27,846 postings keep at least two recognised skills (83.8 percent)
- 165 skills above the rarity filter of 100 postings; 155,027 mentions
- Mean 5.57 skills per posting; deduplicated within each posting

**Speaker notes.** Be explicit that this is a mirror and a three-month window, not the full Kaggle corpus. That is a scope limit, stated up front. The vocabulary decision matters: precision over recall, because a noisy node set corrupts every downstream number.

## Slide 5. Why an edge needs support and selectivity
- Count alone keeps chance pairs: Excel and Leadership meet in any threshold
- PMI compares observed co-occurrence with what popularity predicts
- Chosen: weight >= 50 and PMI >= 0.5, picked from a 15-point sensitivity grid
  - 157 of 165 skills stay connected; moving to t = 100 leaves 34 components
- Result: 883 edges, mean degree 11.25, max degree 53 (Python)

Figure: `figures/fig3_edge_weight_threshold.png`

**Speaker notes.** This is the methodological decision worth the most marks. A count threshold keeps popularity pairs; PMI removes them. Point at the red line on the histogram and say what would happen on either side: too low and the graph is dense and trivial, too high and it shatters into dozens of components.

## Slide 6. Four guards against fooling ourselves
- Hide a random 20 percent of edges: 710 train, 173 test, 19.6 percent
  - Four edges repaired for connectivity and excluded from the test set
- Score all 11,536 non-edges: exact Precision@K, no sampling noise
- Repeat over 25 splits and report mean with standard deviation
- Two negative regimes: uniform random non-edges and distance-2 hard negatives
- Temporal split: train before 2023-11-02, test on pairs seen only afterwards

**Speaker notes.** The exam question is always leakage. Walk through the three answers: features come from the training graph only, the classifier's fitting half is disjoint from its scoring half, and the temporal split removes the possibility that a hidden edge is recoverable from elsewhere in the same corpus.

## Slide 7. Five indices, implemented by hand
- Common Neighbours: |N(u) intersect N(v)|
- Jaccard: intersection / union - penalises large neighbourhoods
- Adamic-Adar: sum 1/log(deg(w)) over shared neighbours
- Preferential Attachment: deg(u) * deg(v), uses no neighbourhood
- Resource Allocation: sum 1/deg(w), a sharper Adamic-Adar
- Cross-checked against NetworkX on 200 pairs: maximum difference 0

**Speaker notes.** Say the formulas, then the cross-check result. The assertion in the code would fail loudly if the hand implementation drifted from NetworkX, which is the answer to 'did you implement it or call a library'. Mention that the graph passed to the indices is unweighted, which is the textbook definition.

## Slide 8. Neighbourhood indices work; degree alone does not

| Method | P@10 | P@20 | P@50 | P@100 | AUC |
|---|---|---|---|---|---|
| Common Neighbours | 0.152 | 0.136 | 0.108 | 0.090 | 0.8711 |
| Jaccard | 0.056 | 0.062 | 0.050 | 0.046 | 0.8478 |
| Adamic-Adar | 0.124 | 0.142 | 0.111 | 0.097 | 0.8819 |
| Preferential Attachment | 0.072 | 0.046 | 0.032 | 0.029 | 0.7787 |
| Resource Allocation | 0.068 | 0.098 | 0.111 | 0.086 | 0.8837 |

*Mean over 25 random splits; 11,536 candidates with 1.5 percent positives. A random ranking would score about 0.015.*

**Speaker notes.** Lead with the base rate, because 0.15 precision sounds weak until it is compared with 0.015. Then the ordering: AA and RA best on AUC, CN best at K=10, Jaccard penalised twice by the union, PA worst. The PA explanation is assortativity (-0.126) and generalist hubs, which is the syllabus link.

## Slide 9. The ordering is right, not just the top of the list
- Best single index: Resource Allocation, AUC 0.8837
- Adamic-Adar close behind at 0.8819, best Precision@20 (0.142)
- Standard deviation around 0.03 across splits
- Easy negatives inflate everything: P@20 rises to 0.98-0.995
- Distance-2 negatives are the honest test: AUC falls to 0.67-0.81
  - Method ordering is stable in every regime

Figure: `figures/fig4_roc_curves.png`

**Speaker notes.** Two points. First, the neighbourhood indices separate from PA. Second, and more important, the absolute numbers depend heavily on the negative distribution, so quote the ordering of methods rather than a single AUC. Mention the hard-negative table here if asked.

## Slide 10. Supervised models and a forward-looking test
- Random Forest AUC 0.9189, the best of anything tested
  - Adamic-Adar and Resource Allocation carry 48 percent of importance
- Logistic Regression AUC 0.8894, best P@20 (0.150)
- Temporal split, 100 forward-looking positives:
  - CN P@20 0.45, AUC 0.878
  - Adamic-Adar P@20 0.35, RA 0.35
  - Jaccard collapses: P@20 0.00, AUC 0.655

**Speaker notes.** The forest buys about three points of AUC over the best heuristic but needs labelled pairs, so the heuristics stay the defensible core. Then the temporal slide-within-a-slide: the neighbourhood signal survives forward in time, Jaccard does not, and PA has a decent AUC with no precision at the top - which is why both metrics are shown.

## Slide 11. Seven communities, only partly aligned with job domains
- Louvain: 7 communities, modularity 0.528
- NMI against 19 curated domains: 0.499
- Design and healthcare are pure clusters (80-100 percent own domain)
- The 56-skill software cluster mixes programming, cloud, data and delivery
- Domains are labels; communities are what employers actually bundle

Figure: `figures/fig1_network_communities.png`

**Speaker notes.** The key line is the last bullet: a low domain share inside the big communities is not a failure of the algorithm, it is the finding. Employers bundle across domains. Point at the labelled clusters and name two or three.

## Slide 12. Small-world, homophilous, disassortative
- Clustering 0.5145 against 0.0726 for a random graph
- Average path length 2.74 steps; small-world sigma 6.06
- Transitivity 0.3874: the triadic closure that makes CN work
- Homophily: 0.752 of observed edges inside a community, baseline 0.191
- Degree assortativity -0.126: why Preferential Attachment fails
- Core: k-core 12 - Python, Agile, SQL, AWS, Azure, Java, CI/CD

Figure: `figures/fig2_degree_distribution.png`

**Speaker notes.** This slide links the structure to the prediction results: clustering explains why CN works, assortativity explains why PA does not, and homophily explains why the recommendations are mostly within-community. The degree tail is heavy but not a clean power law (slope -0.78, R2 0.68), which is honest given 157 nodes.

## Slide 13. Top-20 pairs: mostly bridges between communities
- 11,363 unobserved pairs, 548 survive the filters, Top-20 diversified
- Python + Stakeholder Management: 22 shared neighbours, cross-community
- Excel + Payroll, Accounting + Compliance: finance back office
- C# + REST API, Kubernetes + REST API, Angular + Azure: web and cloud convergence
- 55% of predicted pairs are same-community, against 75% observed

Figure: `figures/fig6_top20_predicted_pairs.png`

**Speaker notes.** Explain the filters first (5 shared neighbours, no generic endpoint, minimum degree 5, diversity cap 2), because they are why the list is readable. Then the bridging point: the model proposes more cross-community pairs than the network contains, which is either the market moving or the model extrapolating - and say which evidence would settle it.

## Slide 14. The list, and what to learn next

| # | Skill A | Skill B | Score | CN |
|---|---|---|---|---|
| 1 | Python | Stakeholder Management | 1.540 | 22 |
| 2 | Agile | Data Analysis | 1.090 | 18 |
| 3 | Microsoft Excel | Payroll | 1.038 | 12 |
| 4 | Accounting | Compliance | 0.958 | 9 |
| 5 | Compliance | Quality Assurance | 0.852 | 6 |
| 6 | Quality Assurance | Stakeholder Management | 0.779 | 14 |
| 7 | Microsoft Excel | Process Improvement | 0.735 | 10 |
| 8 | CSS | Python | 0.676 | 12 |
| 9 | Agile | Process Improvement | 0.642 | 9 |
| 10 | Project Management | SQL | 0.604 | 9 |

*recommend('Python') -> Stakeholder Management 1.54, CSS 0.68, Angular 0.54, Project Management 0.39. recommend('Nursing') -> no pair passes the filters.*

**Speaker notes.** Read three rows aloud and interpret them: Python to stakeholder management is the hybrid-role story, Excel to Payroll is the finance story, C# to REST API is the full-stack story. Then the empty Nursing answer: the filters refuse to invent advice, which is a feature of an evidence-based recommender.

## Slide 15. What this establishes, and what it does not
- Neighbourhood link prediction works on skill co-occurrence, and survives a temporal shift
- The choice of index matters; the choice of negative distribution matters more
- Co-occurrence is not complementarity: the graph records bundles, not causation
- Three-month mirror sample, 165-skill vocabulary, 173 test edges per split
- Next: posting-level features (seniority, salary), a larger corpus, forward validation
- Takeaway: the useful recommendations are the cross-community ones

**Speaker notes.** End on the limitation that matters most, then the takeaway, not the other way round. If there is one sentence to leave in the examiner's head: the market's missing links are mostly between technical and business skills, and that is what a learner should act on.
