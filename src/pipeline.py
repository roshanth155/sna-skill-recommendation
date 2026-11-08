"""
Full pipeline for the SNA course project:
  Skill Recommendation via Link Prediction on a skill co-occurrence network.

Stages
  1  load postings + extract skills (curated vocabulary)
  2  clean, deduplicate, rarity filter
  3  build weighted co-occurrence graph, choose edge threshold t, keep giant component
  4  train/test edge split (random 80/20) + negative sampling (+ temporal split)
  5  link-prediction scores: CN, Jaccard, Adamic-Adar, Preferential Attachment, RA
  6  evaluation: Precision@K, Recall, F1, ROC-AUC over many seeds, sampled/hard negatives
  7  supervised comparison: Logistic Regression vs Random Forest
  8  SNA analysis: degree distribution, clustering, communities, homophily, bridges,
     core-periphery, PageRank, HITS, small-world
  9  recommendations: global Top-20 pairs + recommend(skill, k)
 10  exports: GraphML, GEXF, node/edge CSV, interactive pyvis HTML
 11  figures (.png/.svg) and results.json

Run from the project root:  python src/pipeline.py
"""

# %%
# ----------------------------------------------------------------- 0. Setup
import itertools
import json
import math
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd

# make `src` importable when this file is executed as a script
ROOT = Path.cwd()
if (ROOT / "src").exists():
    sys.path.insert(0, str(ROOT))
else:  # notebook opened from notebooks/
    ROOT = Path("..").resolve()
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

from src.skills_vocab import DOMAIN_OF, GENERIC_SKILLS, SKILLS, extract_skills

SEED = 42
CONFIG = {
    "min_postings": 100,        # a skill must appear in >= 100 postings
    "edge_threshold": 50,       # an edge needs >= 50 co-occurring postings
    "pmi_min": 0.5,             # ...and co-occurrence above what popularity predicts
    "test_frac": 0.20,          # 80/20 edge split
    "n_seeds": 25,              # repeated splits for mean +- std
    "k_values": [10, 20, 50, 100],
    "seed": SEED,
    "temporal_support": 5,      # min postings for a temporal test edge
}

DATA = ROOT / "data"
RAW, PROC = DATA / "raw", DATA / "processed"
FIG, GRAPH_DIR, VIZ = ROOT / "figures", ROOT / "graph", ROOT / "viz"
for d in (PROC, FIG, GRAPH_DIR, VIZ):
    d.mkdir(parents=True, exist_ok=True)

random.seed(SEED)
np.random.seed(SEED)
print("config:", CONFIG)

# %%
# ------------------------------------------------------- 1. Load postings
# The pipeline accepts either input layout:
#   (a) Kaggle "skills table"  : job_skills.csv / jobs/job_skills.csv with (job_id, skills)
#   (b) postings with a free-text description (what we use here)
def load_source():
    """Return (postings_df, mode). mode is 'skills_table' or 'descriptions'."""
    skills_file = next((p for p in [RAW / "job_skills.csv", RAW / "jobs" / "job_skills.csv"]
                        if p.exists()), None)
    if skills_file is not None:
        df = pd.read_csv(skills_file, dtype=str)
        return df, "skills_table"

    for name in ["job_postings_hf.csv", "postings.csv", "job_postings.csv"]:
        p = RAW / name
        if p.exists():
            df = pd.read_csv(p, dtype=str)
            return df, "descriptions"
    raise SystemExit(f"no input file in {RAW}. See README.md for the download step.")


raw, MODE = load_source()
print(f"mode={MODE}  shape={raw.shape}")
print("columns:", list(raw.columns))

# %%
# ------------------------------------------------- 2. Skills + cleaning
t0 = time.time()

if MODE == "skills_table":
    # (a) already-structured skills: one row per (job, skill phrase), maybe comma packed
    id_col = next(c for c in raw.columns if c.lower() in ("job_id", "job_link", "id"))
    skill_col = next(c for c in raw.columns if "skill" in c.lower())
    tmp = raw[[id_col, skill_col]].dropna().copy()
    tmp.columns = ["job_id", "value"]
    tmp["value"] = tmp["value"].astype(str).str.split(r"\s*[,|;]\s*")
    tmp = tmp.explode("value")
    from src.skills_vocab import LOOKUP, tokenize

    def canonicalise(text: str):
        # map a free-text skill phrase onto the curated vocabulary, if it is in it
        toks = tuple(tokenize(text))
        return LOOKUP.get(toks) if toks else None

    tmp["skill"] = tmp["value"].map(canonicalise)
    long = tmp.dropna(subset=["skill"])[["job_id", "skill"]]
    meta = pd.DataFrame({"job_id": raw[id_col].dropna().unique()})
else:
    # (b) free text: extract with the curated vocabulary
    id_col = "job_id" if "job_id" in raw.columns else raw.columns[0]
    text_col = "description" if "description" in raw.columns else None
    if text_col is None:
        raise SystemExit("no description column found")

    keep_cols = [c for c in [id_col, "title", "listed_time", "original_listed_time",
                             "formatted_experience_level", "location"] if c in raw.columns]
    meta = raw[keep_cols].copy()
    rows = []
    for jid, text in zip(raw[id_col], raw[text_col]):
        found = extract_skills(text)
        if len(found) >= 2:                      # a 1-skill posting cannot form an edge
            rows.extend((jid, s) for s in found)
    long = pd.DataFrame(rows, columns=["job_id", "skill"])
    meta = meta[meta[id_col].isin(set(long["job_id"]))]

print(f"extracted in {time.time()-t0:.0f}s")
print(f"mentions={len(long):,}  postings={long['job_id'].nunique():,}  "
      f"skills={long['skill'].nunique()}")

# %%
# Deduplicate within a posting (a posting repeating a skill must count once) and
# keep only skills that clear the rarity threshold.
long = long.drop_duplicates(["job_id", "skill"])
counts = long["skill"].value_counts()
kept = counts[counts >= CONFIG["min_postings"]].index
long = long[long["skill"].isin(kept)].copy()

coverage = long["job_id"].nunique() / raw.shape[0]
print(f"after rarity filter: {long['skill'].nunique()} skills, "
      f"{long['job_id'].nunique():,} postings ({coverage:.1%} of all postings)")
print(f"min postings/skill={long['skill'].value_counts().min()}, "
      f"max={long['skill'].value_counts().max()}")

skills_per_posting = long.groupby("job_id")["skill"].nunique()
print("skills per posting:", skills_per_posting.describe()[["mean", "50%", "max"]].to_dict())

long.to_csv(PROC / "skills_long.csv.gz", index=False, compression="gzip")

# %%
# ------------------------------------------- 3. Weighted co-occurrence graph
def pair_counter(df: pd.DataFrame) -> Counter:
    """Count co-occurrences: every posting adds 1 to each pair of its skills."""
    pc: Counter = Counter()
    for _, group in df.groupby("job_id"):
        pc.update(itertools.combinations(sorted(set(group["skill"])), 2))
    return pc


t0 = time.time()
pairs = pair_counter(long)
print(f"{len(pairs):,} distinct co-occurring pairs, max weight {max(pairs.values())}, "
      f"built in {time.time()-t0:.0f}s")

weights = pd.Series(list(pairs.values()))
print("edge-weight quantiles:", {q: int(weights.quantile(q)) for q in (0.5, 0.9, 0.95, 0.99)})

# Pointwise mutual information: how much more often do u and v appear together than
# independence would predict? A count threshold alone keeps "Excel + Leadership"
# (both in thousands of postings, so they meet by chance) and drops "Docker + Jenkins"
# (both rarer). PMI makes an edge mean *selective* co-occurrence, not popularity.
N_POSTINGS = int(long["job_id"].nunique())
skill_counts = long["skill"].value_counts().to_dict()
PMI = {(u, v): math.log((w * N_POSTINGS) / (skill_counts[u] * skill_counts[v]))
       for (u, v), w in pairs.items()}
pmi_series = pd.Series(list(PMI.values()))
print("PMI quantiles:", {q: round(float(pmi_series.quantile(q)), 2)
                         for q in (0.1, 0.5, 0.9)})

# %%
def build_graph(threshold: int, pmi_min: float) -> nx.Graph:
    """Edge rule: enough support (weight >= t) AND enough selectivity (PMI >= pmi_min)."""
    g = nx.Graph()
    g.add_nodes_from(kept)
    for (u, v), w in pairs.items():
        if w >= threshold and PMI[(u, v)] >= pmi_min:
            g.add_edge(u, v, weight=w, pmi=round(PMI[(u, v)], 3))
    return g


# Threshold choice, from the data rather than a guess: t must produce a network that
# is sparse enough for link prediction to be non-trivial, still connected, and whose
# edges reflect selective pairings. The grid makes the trade-off explicit.
sensitivity = []
for t in [30, 50, 75, 100, 150]:
    for pmi_min in [0.0, 0.5, 1.0]:
        Gt = build_graph(t, pmi_min)
        comps = sorted(nx.connected_components(Gt), key=len, reverse=True)
        giant = Gt.subgraph(comps[0])
        sensitivity.append({
            "t": t, "pmi_min": pmi_min, "nodes": Gt.number_of_nodes(),
            "edges": Gt.number_of_edges(), "components": len(comps),
            "giant": giant.number_of_nodes(), "density": round(nx.density(Gt), 4),
            "transitivity": round(nx.transitivity(Gt), 4),
            "diameter_giant": nx.diameter(giant) if nx.is_connected(giant) else None,
        })
sensitivity_df = pd.DataFrame(sensitivity)
print(sensitivity_df.to_string(index=False))

T = CONFIG["edge_threshold"]
PMI_MIN = CONFIG["pmi_min"]
# %%
G_full = build_graph(T, PMI_MIN)

components = sorted(nx.connected_components(G_full), key=len, reverse=True)
G = G_full.subgraph(components[0]).copy()          # keep the giant component
print(f"t={T}, pmi>={PMI_MIN}: full graph n={G_full.number_of_nodes()} "
      f"m={G_full.number_of_edges()}, {len(components)} components")
print(f"giant component: n={G.number_of_nodes()} m={G.number_of_edges()} "
      f"density={nx.density(G):.4f} "
      f"avg_degree={2*G.number_of_edges()/G.number_of_nodes():.1f} "
      f"transitivity={nx.transitivity(G):.4f}")
print(f"dropped as isolated: {sorted(set(G_full) - set(G))}")

# %%
# ------------------------------------- 4. Split + negative sampling
def split_train_test(G: nx.Graph, test_frac: float, seed: int):
    """Hide a random `test_frac` share of edges; repair connectivity if needed.

    Returns (train_graph, test_edges, repaired_count). Repairing matters: a 20%
    random removal can cut a leaf skill off the graph, and then its edges could not
    be scored at all. Edges put back are excluded from the test set, so no leakage.
    """
    edges = [(u, v, d["weight"]) for u, v, d in G.edges(data=True)]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(edges))
    n_test = int(round(test_frac * len(edges)))
    test = [edges[i] for i in order[:n_test]]
    train = [edges[i] for i in order[n_test:]]

    H = nx.Graph()
    H.add_nodes_from(G.nodes)
    H.add_weighted_edges_from(train)

    repaired = 0
    remaining = []
    for u, v, w in test:
        if u not in H or v not in H or not nx.has_path(H, u, v):
            H.add_edge(u, v, weight=w)
            repaired += 1
        else:
            remaining.append((u, v, w))
    return H, remaining, repaired


H, test_edges, repaired = split_train_test(G, CONFIG["test_frac"], SEED)
print(f"train: n={H.number_of_nodes()} m={H.number_of_edges()} "
      f"| test: {len(test_edges)} edges | repaired for connectivity: {repaired}")
print("test share:", round(len(test_edges) / G.number_of_edges(), 3))

# %%
# Candidate set: every pair that is NOT an edge in the train graph. Enumerating all
# of them is possible because the graph is small (155 nodes -> ~12k pairs), which
# makes Precision@K an exact measurement instead of a sample estimate.
nodes = sorted(G.nodes)
all_pairs = [(u, v) for u, v in itertools.combinations(nodes, 2)]
train_edge_set = {frozenset(e) for e in H.edges}
candidates = [(u, v) for u, v in all_pairs if frozenset((u, v)) not in train_edge_set]
test_set = {frozenset((u, v)) for u, v, _ in test_edges}
y_true = np.array([1 if frozenset(p) in test_set else 0 for p in candidates])
print(f"candidates={len(candidates):,}  positives={y_true.sum()}  "
      f"prevalence={y_true.mean():.4f}")

# %%
# Negative sampling, two variants.
#   easy : uniform random non-edges  (the standard benchmark setting)
#   hard : pairs at distance exactly 2 in the train graph (share a neighbour but are
#          not connected) - the realistic "should this be an edge?" question.
def sample_negatives(H, G, positives, n, seed, hard=False):
    rng = random.Random(seed)
    if not hard:
        non_edges = [p for p in all_pairs
                     if p not in train_edge_set and frozenset(p) not in test_set]
        return rng.sample(non_edges, min(n, len(non_edges)))
    neigh = {n_: set(H.neighbors(n_)) for n_ in H}
    dist2 = [(u, v) for u, v in all_pairs
             if v not in neigh[u] and (neigh[u] & neigh[v])]
    return rng.sample(dist2, min(n, len(dist2)))


neg_easy = sample_negatives(H, G, test_edges, len(test_edges), SEED)
neg_hard = sample_negatives(H, G, test_edges, len(test_edges), SEED, hard=True)
print(f"negatives easy={len(neg_easy)}  hard(distance-2)={len(neg_hard)}")


# %%
# ---------------------------------------------------- 5. Prediction scores
# Implemented by hand from the formulas (viva requirement) and cross-checked
# against NetworkX on a sample.
def common_neighbours(H, u, v):
    return H[u].keys() & H[v].keys()


def score_cn(H, u, v):
    return len(common_neighbours(H, u, v))


def score_jaccard(H, u, v):
    cn = len(common_neighbours(H, u, v))
    union = H.degree(u) + H.degree(v) - cn
    return cn / union if union else 0.0


def score_aa(H, u, v):
    """Adamic-Adar: rare shared neighbours count more (1/log degree)."""
    return sum(1.0 / math.log(H.degree(w)) for w in common_neighbours(H, u, v)
               if H.degree(w) > 1)


def score_pa(H, u, v):
    """Preferential attachment: product of degrees (no neighbourhood needed)."""
    return H.degree(u) * H.degree(v)


def score_ra(H, u, v):
    """Resource allocation: 1/degree, a sharper Adamic-Adar."""
    return sum(1.0 / H.degree(w) for w in common_neighbours(H, u, v))


METHODS = {
    "CN": score_cn,
    "Jaccard": score_jaccard,
    "Adamic-Adar": score_aa,
    "Preferential Attachment": score_pa,
    "Resource Allocation": score_ra,
}

# Cross-check the hand-written implementations against NetworkX.
sample_pairs = candidates[:200]
check = {
    "CN": max(abs(score_cn(H, u, v) - len(list(nx.common_neighbors(H, u, v))))
              for u, v in sample_pairs),
    "Jaccard": max(abs(score_jaccard(H, u, v) - p)
                   for (u, v), (_, _, p) in zip(sample_pairs,
                                                nx.jaccard_coefficient(H, sample_pairs))),
    "Adamic-Adar": max(abs(score_aa(H, u, v) - p)
                       for (u, v), (_, _, p) in zip(sample_pairs,
                                                    nx.adamic_adar_index(H, sample_pairs))),
    "PA": max(abs(score_pa(H, u, v) - p)
              for (u, v), (_, _, p) in zip(sample_pairs,
                                           nx.preferential_attachment(H, sample_pairs))),
    "RA": max(abs(score_ra(H, u, v) - p)
              for (u, v), (_, _, p) in zip(sample_pairs,
                                           nx.resource_allocation_index(H, sample_pairs))),
}
print("max abs difference vs NetworkX:", {k: round(v, 10) for k, v in check.items()})
assert all(v < 1e-9 for v in check.values())

# %%
def rank_metrics(scores: np.ndarray, labels: np.ndarray, k_values) -> dict:
    """Precision@K, Recall@K, F1@K from one score vector over a candidate set."""
    order = np.argsort(-scores)
    out = {}
    n_pos = int(labels.sum())
    for k in k_values:
        topk = order[:k]
        tp = int(labels[topk].sum())
        precision = tp / k
        recall = tp / n_pos if n_pos else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if precision + recall else 0.0
        out[f"P@{k}"] = precision
        out[f"R@{k}"] = recall
        out[f"F1@{k}"] = f1
    return out


def auc_score(scores: np.ndarray, labels: np.ndarray) -> float:
    if labels.sum() == 0 or labels.sum() == len(labels):
        return float("nan")
    return float(roc_auc_score(labels, scores))


# %%
# ------------------------------------------------ 6. Evaluation over seeds
from sklearn.metrics import roc_auc_score, roc_curve

results_per_seed = {m: [] for m in METHODS}
roc_curves = {}
t0 = time.time()

for seed in range(CONFIG["n_seeds"]):
    H_s, test_s, _ = split_train_test(G, CONFIG["test_frac"], seed)
    ts = {frozenset((u, v)) for u, v, _ in test_s}
    labels_s = np.array([1 if frozenset(p) in ts else 0 for p in candidates])

    for name, fn in METHODS.items():
        sc = np.array([fn(H_s, u, v) for u, v in candidates])
        met = rank_metrics(sc, labels_s, CONFIG["k_values"])
        met["AUC"] = auc_score(sc, labels_s)
        results_per_seed[name].append(met)

    if seed == 0:            # one fixed split is kept for the ROC figure
        for name, fn in METHODS.items():
            sc = np.array([fn(H_s, u, v) for u, v in candidates])
            fpr, tpr, _ = roc_curve(labels_s, sc)
            idx = np.linspace(0, len(fpr) - 1, 200).astype(int)
            roc_curves[name] = {"fpr": fpr[idx].round(5).tolist(),
                                "tpr": tpr[idx].round(5).tolist(),
                                "auc": float(np.trapezoid(tpr, fpr))}

print(f"seed loop: {time.time()-t0:.0f}s for {CONFIG['n_seeds']} seeds")

summary_rows = []
for name, runs in results_per_seed.items():
    dfm = pd.DataFrame(runs)
    row = {"method": name}
    for col in dfm.columns:
        row[col] = round(dfm[col].mean(), 4)
        row[col + "_std"] = round(dfm[col].std(ddof=1), 4)
    summary_rows.append(row)
summary_df = pd.DataFrame(summary_rows)
print(summary_df[["method", "P@10", "P@20", "P@50", "P@100", "AUC"]].to_string(index=False))
summary_df.to_csv(PROC / "metrics_by_method.csv", index=False)

# %%
# Same models, but ranked against 1:1 negatives (the usual benchmark setting) and
# against distance-2 hard negatives. Reporting all three shows how much of the
# performance comes from the easy negative distribution.
def evaluate_on_pair_set(H_s, positives, negatives, k=20):
    pairs = list(positives) + list(negatives)
    labels = np.array([1] * len(positives) + [0] * len(negatives))
    out = {}
    for name, fn in METHODS.items():
        sc = np.array([fn(H_s, u, v) for u, v in pairs])
        out[name] = {"AUC": auc_score(sc, labels),
                     f"P@{k}": float(labels[np.argsort(-sc)[:k]].sum() / k)}
    return out


easy_metrics = {m: [] for m in METHODS}
hard_metrics = {m: [] for m in METHODS}
for seed in range(min(10, CONFIG["n_seeds"])):
    H_s, test_s, _ = split_train_test(G, CONFIG["test_frac"], seed)
    pos = [(u, v) for u, v, _ in test_s]
    rng = random.Random(seed + 1000)
    all_pairs_s = [(u, v) for u, v in itertools.combinations(sorted(H_s.nodes), 2)]
    ts = {frozenset(p) for p in pos}
    non_edges = [p for p in all_pairs_s if frozenset(p) not in ts
                 and not H_s.has_edge(*p)]
    neg_e = rng.sample(non_edges, min(len(pos), len(non_edges)))
    neigh = {n_: set(H_s.neighbors(n_)) for n_ in H_s}
    dist2 = [(u, v) for u, v in non_edges if v not in neigh[u] and (neigh[u] & neigh[v])]
    neg_h = rng.sample(dist2, min(len(pos), len(dist2))) if dist2 else []

    for src, res in ((evaluate_on_pair_set(H_s, pos, neg_e), easy_metrics),
                     (evaluate_on_pair_set(H_s, pos, neg_h) if neg_h else {}, hard_metrics)):
        for name, vals in src.items():
            res[name].append(vals)

sampled_table = pd.DataFrame([
    {"method": m,
     "AUC_1to1": round(np.mean([r["AUC"] for r in easy_metrics[m]]), 4),
     "P@20_1to1": round(np.mean([r["P@20"] for r in easy_metrics[m]]), 4),
     "AUC_hard": round(np.mean([r["AUC"] for r in hard_metrics[m]]), 4) if hard_metrics[m] else None,
     "P@20_hard": round(np.mean([r["P@20"] for r in hard_metrics[m]]), 4) if hard_metrics[m] else None}
    for m in METHODS])
print(sampled_table.to_string(index=False))
sampled_table.to_csv(PROC / "metrics_sampled_negatives.csv", index=False)

# %%
# ------------------------------------------------- 7. Supervised comparison
# Leakage control: features come ONLY from the train graph. The held-out positives
# are split in half: one half fits the classifier, the other half is the test set.
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def pair_features(H_s, pairs):
    feats = []
    for u, v in pairs:
        cn = len(common_neighbours(H_s, u, v))
        feats.append([cn, score_jaccard(H_s, u, v), score_aa(H_s, u, v),
                      score_pa(H_s, u, v), score_ra(H_s, u, v),
                      min(H_s.degree(u), H_s.degree(v)),
                      max(H_s.degree(u), H_s.degree(v))])
    return np.asarray(feats, dtype=float)


FEATURE_NAMES = ["CN", "Jaccard", "AA", "PA", "RA", "min_degree", "max_degree"]
ml_results = {"Logistic Regression": [], "Random Forest": []}
rf_for_importance = None

# Fair comparison: the classifier is fitted on a dev half of the held-out edges,
# then scored on exactly the same full candidate set as the heuristics (not on an
# easier 1:1 subset), so the table compares like with like.
for seed in range(min(10, CONFIG["n_seeds"])):
    H_s, test_s, _ = split_train_test(G, CONFIG["test_frac"], seed)
    pos = [(u, v) for u, v, _ in test_s]
    ts = {frozenset(p) for p in pos}
    labels_s = np.array([1 if frozenset(p) in ts else 0 for p in candidates])

    rng = random.Random(seed + 500)
    all_pairs_s = [(u, v) for u, v in itertools.combinations(sorted(H_s.nodes), 2)]
    non_edges = [p for p in all_pairs_s if frozenset(p) not in ts and not H_s.has_edge(*p)]
    pos_fit, _pos_unused = train_test_split(pos, test_size=0.5, random_state=seed)
    neg_fit = rng.sample(non_edges, min(len(pos_fit), len(non_edges)))

    X_fit = pair_features(H_s, pos_fit + neg_fit)
    labels_fit = np.array([1] * len(pos_fit) + [0] * len(neg_fit))
    X_all = pair_features(H_s, candidates)

    lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=seed))
    rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                random_state=seed, n_jobs=-1)
    for name, model in (("Logistic Regression", lr), ("Random Forest", rf)):
        model.fit(X_fit, labels_fit)
        prob = model.predict_proba(X_all)[:, 1]
        met = rank_metrics(prob, labels_s, CONFIG["k_values"])
        met["AUC"] = auc_score(prob, labels_s)
        ml_results[name].append(met)
        if seed == 0 and name == "Random Forest":
            rf_for_importance = model

ml_table = pd.DataFrame([
    {"model": name,
     "AUC": round(np.mean([r["AUC"] for r in runs]), 4),
     "AUC_std": round(np.std([r["AUC"] for r in runs], ddof=1), 4),
     "P@20": round(np.mean([r["P@20"] for r in runs]), 4),
     "P@50": round(np.mean([r["P@50"] for r in runs]), 4)}
    for name, runs in ml_results.items()])
print(ml_table.to_string(index=False))
ml_table.to_csv(PROC / "metrics_ml.csv", index=False)

FEATURE_IMPORTANCE = (
    dict(zip(FEATURE_NAMES, rf_for_importance.feature_importances_.round(4).tolist()))
    if rf_for_importance is not None else {})
print("random-forest feature importance:", FEATURE_IMPORTANCE)

# %%
# ------------------------------------------------------- 8. Temporal split
# Second validation: train on the older postings, test on pairs that only appear in
# the newer ones. This is the strongest guard against leakage, because the test
# edges are genuinely future observations rather than randomly hidden ones.
temporal = {}
if MODE == "descriptions" and "listed_time" in meta.columns:
    dates = pd.to_datetime(pd.to_numeric(meta["listed_time"], errors="coerce"),
                           unit="ms", errors="coerce", utc=True)
    meta = meta.assign(date=dates)
    cutoff = dates.quantile(0.5)
    early_ids = set(meta.loc[meta["date"] < cutoff, "job_id"])
    late_ids = set(meta.loc[meta["date"] >= cutoff, "job_id"])
    early = long[long["job_id"].isin(early_ids)]

    # Both windows must apply the same evidence standard. The newer half has roughly
    # the same number of postings here, but if it did not, a pair could clear the
    # weight threshold simply because more postings were observed. We therefore
    # subsample the later window down to the size of the earlier one before counting.
    early_long_ids = sorted(set(long["job_id"]) & early_ids)
    late_long_ids = sorted(set(long["job_id"]) & late_ids)
    rng_t = random.Random(SEED)
    late_sample = set(rng_t.sample(late_long_ids, min(len(early_long_ids), len(late_long_ids))))
    late = long[long["job_id"].isin(late_sample)]

    G_early = nx.Graph()
    G_early.add_nodes_from(kept)
    G_early.add_weighted_edges_from(
        [(u, v, w) for (u, v), w in pair_counter(early).items() if w >= T])
    if G_early.number_of_edges():
        comps_e = sorted(nx.connected_components(G_early), key=len, reverse=True)
        G_early = G_early.subgraph(comps_e[0]).copy()

    late_pairs = pair_counter(late)
    # same threshold t as the random split: "would have been an edge if it had been
    # observed in the training window, but was not"
    test_pos_t = [(u, v) for (u, v), w in late_pairs.items()
                  if w >= T and u in G_early and v in G_early and not G_early.has_edge(u, v)]

    cand_t = [(u, v) for u, v in itertools.combinations(sorted(G_early.nodes), 2)
              if not G_early.has_edge(u, v)]
    ts_t = {frozenset(p) for p in test_pos_t}
    labels_t = np.array([1 if frozenset(p) in ts_t else 0 for p in cand_t])

    temporal = {"cutoff": str(cutoff), "train_postings": len(early_long_ids),
                "test_postings": len(late_sample),
                "train_edges": G_early.number_of_edges(),
                "test_positives": len(test_pos_t)}
    temporal["methods"] = {}
    for name, fn in METHODS.items():
        sc = np.array([fn(G_early, u, v) for u, v in cand_t])
        met = rank_metrics(sc, labels_t, [20, 100])
        met["AUC"] = auc_score(sc, labels_t)
        temporal["methods"][name] = {k: round(float(v), 4) for k, v in met.items()}
    print("temporal split:", {k: v for k, v in temporal.items() if k != "methods"})
    print(pd.DataFrame(temporal["methods"]).T[["P@20", "AUC"]].to_string())
else:
    print("no timestamps available - temporal validation skipped")

# %%
# -------------------------------------------------------- 9. SNA analysis
degrees = dict(G.degree())
deg_values = np.array(list(degrees.values()))

# degree distribution: log-log fit (a straight line hints at a heavy tail)
deg_counts = Counter(deg_values)
xs = np.array(sorted(deg_counts))
ys = np.array([deg_counts[d] for d in xs])
mask = (xs > 0) & (ys > 0)
slope, intercept = np.polyfit(np.log(xs[mask]), np.log(ys[mask]), 1)
resid = np.log(ys[mask]) - (slope * np.log(xs[mask]) + intercept)
r2 = 1 - (resid ** 2).sum() / ((np.log(ys[mask]) - np.log(ys[mask]).mean()) ** 2).sum()

# communities (Louvain) and their alignment with the curated skill domains
communities = nx.community.louvain_communities(G, weight="weight", seed=SEED)
community_of = {n_: i for i, comm in enumerate(communities) for n_ in comm}
modularity = nx.community.modularity(G, communities, weight="weight")

from sklearn.metrics import normalized_mutual_info_score
nmi_domains = normalized_mutual_info_score(
    [community_of[n_] for n_ in G.nodes],
    [DOMAIN_OF[n_] for n_ in G.nodes])

# small-world: compare clustering and path length with a random graph of same n, m
C_obs = nx.average_clustering(G)
L_obs = nx.average_shortest_path_length(G)
rand_C, rand_L = [], []
for s in range(20):
    R = nx.gnm_random_graph(G.number_of_nodes(), G.number_of_edges(), seed=s)
    rand_C.append(nx.average_clustering(R))
    if nx.is_connected(R):
        rand_L.append(nx.average_shortest_path_length(R))
sigma = (C_obs / np.mean(rand_C)) / (L_obs / np.mean(rand_L)) if rand_L else float("nan")

betweenness = nx.betweenness_centrality(G)
constraint = nx.constraint(G)
pagerank = nx.pagerank(G, weight="weight")
try:
    hubs, authorities = nx.hits(G, max_iter=2000, normalized=True)
except nx.PowerIterationFailedConvergence:      # tiny graphs can misbehave
    hubs = authorities = {n_: 0.0 for n_ in G.nodes}
core = nx.core_number(G)

node_df = pd.DataFrame({
    "skill": list(G.nodes),
    "domain": [DOMAIN_OF[n_] for n_ in G.nodes],
    "degree": [degrees[n_] for n_ in G.nodes],
    "weighted_degree": [sum(d["weight"] for _, _, d in G.edges(n_, data=True)) for n_ in G.nodes],
    "community": [community_of[n_] for n_ in G.nodes],
    "betweenness": [round(betweenness[n_], 6) for n_ in G.nodes],
    "constraint": [round(constraint[n_], 6) for n_ in G.nodes],
    "pagerank": [round(pagerank[n_], 6) for n_ in G.nodes],
    "hits_hub": [round(hubs[n_], 6) for n_ in G.nodes],
    "hits_authority": [round(authorities[n_], 6) for n_ in G.nodes],
    "kcore": [core[n_] for n_ in G.nodes],
    "generic": [n_ in GENERIC_SKILLS for n_ in G.nodes],
}).sort_values("degree", ascending=False).reset_index(drop=True)

print("\ntop 10 by degree:\n", node_df.head(10)[["skill", "degree", "community", "domain"]].to_string(index=False))
print("\ntop 8 by betweenness:\n", node_df.sort_values("betweenness", ascending=False)
      .head(8)[["skill", "betweenness", "degree"]].to_string(index=False))
print("\ntop 8 by pagerank:\n", node_df.sort_values("pagerank", ascending=False)
      .head(8)[["skill", "pagerank", "degree"]].to_string(index=False))

community_summary = []
for i, comm in enumerate(communities):
    dom = Counter(DOMAIN_OF[n_] for n_ in comm).most_common(3)
    community_summary.append({
        "community": i, "size": len(comm), "dominant_domain": dom[0][0],
        "domain_share": round(dom[0][1] / len(comm), 3),
        "top_skills": sorted(comm, key=lambda n_: -degrees[n_])[:6],
    })
communities_df = pd.DataFrame(community_summary).sort_values("size", ascending=False)
print("\ncommunities:\n", communities_df.to_string(index=False))

sna = {
    "n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges(),
    "density": round(nx.density(G), 4),
    "avg_degree": round(float(deg_values.mean()), 2),
    "max_degree": int(deg_values.max()),
    "degree_slope_loglog": round(float(slope), 3),
    "degree_fit_r2": round(float(r2), 3),
    "avg_clustering": round(float(C_obs), 4),
    "random_clustering": round(float(np.mean(rand_C)), 4),
    "avg_path_length": round(float(L_obs), 3),
    "random_path_length": round(float(np.mean(rand_L)), 3),
    "small_world_sigma": round(float(sigma), 2),
    "transitivity": round(float(nx.transitivity(G)), 4),
    "assortativity_degree": round(float(nx.degree_assortativity_coefficient(G)), 3),
    "n_communities": len(communities),
    "modularity": round(float(modularity), 3),
    "nmi_communities_vs_domains": round(float(nmi_domains), 3),
    "max_kcore": int(max(core.values())),
    "core_skills": sorted(G.nodes, key=lambda n_: -core[n_])[:10],
}

# %%
# ------------------------------ 10. Recommendations on the observed graph
# Score every pair that is NOT an edge in the full observed graph: these are the
# "not yet observed together" candidates. The best method from evaluation is used.
best_method = summary_df.sort_values("AUC", ascending=False).iloc[0]["method"]
print("best method by mean AUC:", best_method)

full_pairs = [(u, v) for u, v in itertools.combinations(sorted(G.nodes), 2)
              if not G.has_edge(u, v)]
score_fn = METHODS[best_method]
pred_scores = np.array([score_fn(G, u, v) for u, v in full_pairs])

cand = pd.DataFrame(full_pairs, columns=["skill_a", "skill_b"])
cand["score"] = pred_scores
cand["cn"] = [len(common_neighbours(G, u, v)) for u, v in full_pairs]
cand["deg_a"] = [G.degree(u) for u, v in full_pairs]
cand["deg_b"] = [G.degree(v) for u, v in full_pairs]
cand["cand_deg_min"] = cand[["deg_a", "deg_b"]].min(axis=1)
cand["generic_endpoint"] = (cand["skill_a"].isin(GENERIC_SKILLS)
                            | cand["skill_b"].isin(GENERIC_SKILLS))

# Recommendation filters (the evaluation above is untouched by these):
#   cn >= 5      a pair justified by one shared neighbour is coincidence, not demand
#   no generic   "Git + Leadership" is not an actionable thing to learn next
#   min degree 5 a skill seen in <5 strong pairings cannot support a recommendation
REC_MIN_CN = 5
REC_MIN_DEGREE = 5
ranked = cand[(cand["cn"] >= REC_MIN_CN)
              & (~cand["generic_endpoint"])
              & (cand["cand_deg_min"] >= REC_MIN_DEGREE)]
ranked = ranked.sort_values(["score", "cn"], ascending=False)
print(f"recommendation candidates: {len(cand):,} -> {len(ranked):,} after filters "
      f"(cn>={REC_MIN_CN}, no generic endpoint, min degree >= {REC_MIN_DEGREE})")

def select_diverse(df: pd.DataFrame, k: int = 20, max_per_skill: int = 2) -> pd.DataFrame:
    """Take the top-k pairs, allowing any one skill to appear at most `max_per_skill`
    times. Without this the single strongest hub fills the whole list (in an earlier
    run 'Stakeholder Management' took 5 of the 20 slots), which is useless as advice."""
    seen: Counter = Counter()
    picked = []
    for idx, row in df.iterrows():
        if seen[row["skill_a"]] >= max_per_skill or seen[row["skill_b"]] >= max_per_skill:
            continue
        seen[row["skill_a"]] += 1
        seen[row["skill_b"]] += 1
        picked.append(idx)
        if len(picked) == k:
            break
    return df.loc[picked]


top20 = select_diverse(ranked, k=20, max_per_skill=2).copy()
top20["shared_neighbours"] = [
    ", ".join(sorted(common_neighbours(G, u, v))[:5]) for u, v in zip(top20["skill_a"], top20["skill_b"])]
top20["community_a"] = top20["skill_a"].map(community_of)
top20["community_b"] = top20["skill_b"].map(community_of)
top20["same_community"] = top20["community_a"] == top20["community_b"]
top20["domain_a"] = top20["skill_a"].map(DOMAIN_OF)
top20["domain_b"] = top20["skill_b"].map(DOMAIN_OF)

# homophily baseline: what share of random non-edges sit inside one community?
rng = random.Random(SEED)
baseline = [rng.choice(full_pairs) for _ in range(5000)]
baseline_same = np.mean([community_of[u] == community_of[v] for u, v in baseline])
homophily = {
    "observed_edges_same_community": round(
        float(np.mean([community_of[u] == community_of[v] for u, v in G.edges])), 3),
    "predicted_top20_same_community": round(float(top20["same_community"].mean()), 3),
    "random_non_edge_baseline": round(float(baseline_same), 3),
}

print("\nTop 20 predicted skill pairs (not currently connected):")
print(top20[["skill_a", "skill_b", "score", "cn", "same_community",
             "domain_a", "domain_b"]].to_string(index=False))
print("homophily:", homophily)

# %%
def recommend(skill: str, k: int = 5) -> list[tuple[str, float, int]]:
    """Skills most likely to be demanded together with `skill`, most likely first.

    Uses the same link-prediction score on the observed graph; already-connected
    skills are excluded by construction, and the recommendation filters above
    remove generic and weakly-supported pairs.
    """
    if skill not in G:
        raise KeyError(f"{skill!r} is not in the network")
    pool = ranked[ranked["skill_a"].eq(skill) | ranked["skill_b"].eq(skill)]
    return [(row.skill_a if row.skill_b == skill else row.skill_b, round(row.score, 3), int(row.cn))
            for row in pool.head(k).itertuples()]


for demo_skill in ["Python", "SQL", "Nursing", "Microsoft Excel"]:
    if demo_skill in G:
        recs = recommend(demo_skill, 5)
        print(f"\nlearn next after {demo_skill}:")
        if not recs:
            print("   (no candidate passed the recommendation filters)")
        for name, sc, cn in recs:
            print(f"   {name:28s} score={sc:7.3f} shared_neighbours={cn}")
    else:
        print(f"\n{demo_skill} is not in the giant component")

# %%
# ------------------------------------------------------------ 11. Exports
# Node/edge tables and network files for Cytoscape / Gephi.
edge_rows = []
for u, v, d in G.edges(data=True):
    edge_rows.append({"source": u, "target": v, "weight": d["weight"],
                      "predicted": 0, "score": 0.0})
for row in top20.itertuples():
    edge_rows.append({"source": row.skill_a, "target": row.skill_b, "weight": 0,
                      "predicted": 1, "score": float(row.score)})

edges_df = pd.DataFrame(edge_rows)
node_df.to_csv(PROC / "nodes.csv", index=False)
edges_df.to_csv(PROC / "edges.csv", index=False)
long.to_csv(PROC / "skills_long.csv.gz", index=False, compression="gzip",
            columns=["job_id", "skill"])

# graph files with the attributes attached
Gx = G.copy()
for n_ in Gx.nodes:
    row = node_df.set_index("skill").loc[n_]
    for attr in ["domain", "degree", "community", "betweenness", "constraint",
                 "pagerank", "hits_authority", "kcore"]:
        Gx.nodes[n_][attr] = row[attr]
for u, v, d in Gx.edges(data=True):
    d["predicted"] = 0
    d["score"] = 0.0
for row in top20.itertuples():
    Gx.add_edge(row.skill_a, row.skill_b, weight=0, predicted=1, score=float(row.score))

nx.write_graphml(Gx, GRAPH_DIR / "skill_network.graphml")
nx.write_gexf(Gx, GRAPH_DIR / "skill_network.gexf")
edges_df.to_csv(GRAPH_DIR / "skill_edges_cytoscape.csv", index=False)
node_df.to_csv(GRAPH_DIR / "skill_nodes_cytoscape.csv", index=False)

# interactive HTML of the Top-20 predicted pairs with their graph neighbourhood
try:
    from pyvis.network import Network

    focus = set(top20["skill_a"]) | set(top20["skill_b"])
    # add each focus node's three strongest neighbours so the predicted links
    # are shown in a real neighbourhood, not as floating nodes
    context = set()
    for n_ in focus:
        strongest = sorted(G[n_].items(), key=lambda kv: -kv[1]["weight"])[:3]
        context |= {w for w, _ in strongest}
    SG = G.subgraph(focus | context).copy()

    net = Network(height="750px", width="100%", bgcolor="#ffffff",
                  font_color="#222222", notebook=False, cdn_resources="in_line")
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120)
    for n_ in SG.nodes:
        net.add_node(n_, label=n_,
                     title=f"{n_} | domain={DOMAIN_OF[n_]} | degree={G.degree(n_)}",
                     value=G.degree(n_), group=community_of[n_])
    for u, v, d in SG.edges(data=True):
        net.add_edge(u, v, value=d["weight"], title=f"observed (weight {d['weight']})",
                     color="#c8c8c8", width=1)
    for row in top20.itertuples():
        if row.skill_a in SG and row.skill_b in SG:
            net.add_edge(row.skill_a, row.skill_b, dashes=True, color="#d62728", width=4,
                         title=f"predicted ({best_method} {row.score:.2f})")
    # pyvis writes with the platform default encoding, which fails on Windows for
# non-ASCII characters in the template, so the HTML string is written explicitly.
    html = net.generate_html(notebook=False)
    (VIZ / "top20_predicted_links.html").write_text(html, encoding="utf-8")
    print("wrote", VIZ / "top20_predicted_links.html")
except Exception as exc:  # pyvis is optional
    print("pyvis export skipped:", exc)

# %%
# ----------------------------------------------------------- 12. Figures
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"figure.dpi": 130, "savefig.bbox": "tight",
                     "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", dpi=200)
    fig.savefig(FIG / f"{name}.svg")
    if "agg" not in matplotlib.get_backend().lower():
        plt.show()      # renders inline in the notebook, no-op with the Agg backend
    plt.close(fig)
    print("figure:", name)


# (1) the network itself
# A plain spring layout turns 157 nodes into a hairball with colliding labels, so
# communities are laid out as separate clusters around a ring: the structure of the
# graph is readable at a glance and each group can be labelled once.
def community_layout(G, community_of, seed=0):
    rng = np.random.default_rng(seed)
    comms = sorted({c for c in community_of.values()},
                   key=lambda c: -sum(1 for n in G if community_of[n] == c))
    sizes = {c: sum(1 for n in G if community_of[n] == c) for c in comms}
    total_weight = sum(math.sqrt(s) for s in sizes.values())
    pos, angle = {}, 0.0
    for c in comms:
        share = 2 * math.pi * math.sqrt(sizes[c]) / total_weight
        centre_angle = angle + share / 2
        orbit = 2.2 + 0.42 * math.sqrt(len(G))          # ring of clusters
        cx, cy = orbit * math.cos(centre_angle), orbit * math.sin(centre_angle)
        members = sorted([n for n in G if community_of[n] == c], key=lambda n: -G.degree(n))
        blob = 0.55 * math.sqrt(sizes[c])               # cluster radius grows with size
        for i, n in enumerate(members):
            # golden-angle spiral: even spacing, no overlaps, no randomness
            r = blob * math.sqrt((i + 0.5) / len(members))
            theta = i * 2.399963229728653
            pos[n] = (cx + r * math.cos(theta), cy + r * math.sin(theta))
        angle += share
    return pos


pos = community_layout(G, community_of, seed=SEED)
palette = plt.get_cmap("tab20")
fig, ax = plt.subplots(figsize=(12, 9))
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.07, width=0.4, edge_color="#333333")
node_sizes = np.array([degrees[n_] for n_ in G.nodes]) * 3.2
colors = [palette(community_of[n_] * 2 % 20) for n_ in G.nodes]
nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_sizes, node_color=colors,
                       linewidths=0.5, edgecolors="white")
labels = {n_: n_ for n_ in G.nodes if degrees[n_] >= np.quantile(deg_values, 0.70)}
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax, font_size=6.2,
                        bbox=dict(facecolor="white", alpha=0.6, edgecolor="none", pad=0.1))
# one title per cluster instead of a legend the eye has to map back to colours
for c, comm in enumerate(communities):
    members = [n_ for n_ in comm if n_ in pos]
    if not members:
        continue
    cx = np.mean([pos[n_][0] for n_ in members])
    cy = max(pos[n_][1] for n_ in members) + 0.42
    dom = Counter(DOMAIN_OF[n_] for n_ in members).most_common(1)[0][0]
    ax.text(cx, cy, f"C{c}: {dom} ({len(members)})", ha="center", va="bottom",
            fontsize=8.5, fontweight="bold", color=palette(c * 2 % 20),
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none", pad=0.2))
ax.set_title(f"Skill co-occurrence network (n={G.number_of_nodes()}, "
             f"m={G.number_of_edges()}, {len(communities)} communities)")
ax.axis("off")
save(fig, "fig1_network_communities")

# (2) degree distribution
fig, ax = plt.subplots(figsize=(6, 4.2))
ax.loglog(xs, ys, "o", ms=4, alpha=0.8)
fit_x = np.linspace(xs[mask].min(), xs[mask].max(), 50)
ax.loglog(fit_x, np.exp(intercept) * fit_x ** slope, "--",
          label=f"log-log fit: slope={slope:.2f}, $R^2$={r2:.2f}")
ax.set_xlabel("degree k")
ax.set_ylabel("number of skills with degree k")
ax.set_title("Degree distribution")
ax.legend()
save(fig, "fig2_degree_distribution")

# (3) edge-weight distribution and the threshold
fig, ax = plt.subplots(figsize=(6, 4))
ax.hist(weights.clip(upper=1000), bins=60)
ax.axvline(T, color="crimson", ls="--", label=f"threshold t={T}")
ax.set_yscale("log")
ax.set_xlabel("edge weight (co-occurring postings)")
ax.set_ylabel("number of pairs (log)")
ax.set_title("Co-occurrence weight distribution")
ax.legend()
save(fig, "fig3_edge_weight_threshold")

# (4) ROC curves
fig, ax = plt.subplots(figsize=(5.6, 4.6))
for name, curve in roc_curves.items():
    ax.plot(curve["fpr"], curve["tpr"], lw=1.4,
            label=f"{name} (AUC={curve['auc']:.3f})")
ax.plot([0, 1], [0, 1], "k--", lw=0.8)
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_title("ROC curves on the held-out 20% of edges")
ax.legend(fontsize=7)
save(fig, "fig4_roc_curves")

# (5) metric comparison with error bars
fig, ax = plt.subplots(figsize=(7.5, 4.2))
metric_names = ["AUC", "P@20", "P@50"]
x = np.arange(len(metric_names))
width = 0.16
for i, (name, runs) in enumerate(results_per_seed.items()):
    dfm = pd.DataFrame(runs)
    means = [dfm[m].mean() for m in metric_names]
    errs = [dfm[m].std(ddof=1) for m in metric_names]
    ax.bar(x + i * width, means, width, yerr=errs, label=name, capsize=2)
ax.set_xticks(x + width * 2)
ax.set_xticklabels(metric_names)
ax.set_ylabel("score (mean over seeds)")
ax.set_title(f"Heuristic comparison ({CONFIG['n_seeds']} random splits)")
ax.legend(fontsize=7)
save(fig, "fig5_metric_comparison")

# (6) Top-20 predicted pairs as a subgraph
# (6) Top-20 predicted pairs as their own subgraph, labelled so the figure is
# readable on a slide: only the endpoints are shown, observed edges between them
# in grey, predicted (missing) edges in dashed red.
pred_nodes = sorted(set(top20["skill_a"]) | set(top20["skill_b"]))
sub = G.subgraph(pred_nodes).copy()
pos2 = nx.spring_layout(sub, seed=SEED, k=1.3, iterations=600)
fig, ax = plt.subplots(figsize=(10.5, 7))
nx.draw_networkx_edges(sub, pos2, ax=ax, alpha=0.35, width=1.0, edge_color="#999999")
nx.draw_networkx_nodes(sub, pos2, ax=ax, node_size=260,
                       node_color=[palette(community_of[n_] * 2 % 20) for n_ in sub.nodes],
                       linewidths=0.6, edgecolors="white")
nx.draw_networkx_labels(sub, pos2, ax=ax, font_size=7.5,
                        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.2))
for row in top20.itertuples():
    ax.plot([pos2[row.skill_a][0], pos2[row.skill_b][0]],
            [pos2[row.skill_a][1], pos2[row.skill_b][1]],
            color="crimson", lw=2.2, ls=(0, (4, 2)), alpha=0.9)
ax.set_title(f"Top-20 predicted skill pairs (dashed red) - {best_method}")
ax.axis("off")
save(fig, "fig6_top20_predicted_pairs")

# (7) k-core / core-periphery profile
fig, ax = plt.subplots(figsize=(6.4, 3.8))
core_counts = pd.Series(core).value_counts().sort_index()
ax.bar(core_counts.index, core_counts.values)
ax.set_xlabel("k-core shell")
ax.set_ylabel("number of skills")
ax.set_title("Core-periphery structure (k-core decomposition)")
save(fig, "fig7_kcore_profile")

# %%
# ------------------------------------------------------ 13. Save results
top20_out = top20[["skill_a", "skill_b", "score", "cn", "same_community",
                   "domain_a", "domain_b", "shared_neighbours"]].copy()
top20_out["score"] = top20_out["score"].round(3)
top20_out.to_csv(PROC / "predictions_top20.csv", index=False)
ranked.head(200)[["skill_a", "skill_b", "score", "cn"]].round(3).to_csv(
    PROC / "predictions_top200.csv", index=False)

results = {
    "config": CONFIG,
    "dataset": {
        "source_file": "data/raw/job_postings_hf.csv",
        "mode": MODE,
        "postings_raw": int(raw.shape[0]),
        "skills_vocabulary": len(SKILLS),
        "skills_kept": int(long["skill"].nunique()),
        "postings_used": int(long["job_id"].nunique()),
        "posting_coverage": round(float(coverage), 4),
        "mentions": int(len(long)),
        "mean_skills_per_posting": round(float(skills_per_posting.mean()), 2),
        "date_min": str(meta["date"].min()) if "date" in meta else None,
        "date_max": str(meta["date"].max()) if "date" in meta else None,
    },
    "graph": {"threshold": T, "pmi_min": PMI_MIN, "sensitivity": sensitivity,
              "n_nodes_full": G_full.number_of_nodes(),
              "n_edges_full": G_full.number_of_edges(),
              "components_full": len(components),
              "isolated_dropped": sorted(set(G_full) - set(G))},
    "split": {"train_edges": H.number_of_edges(), "test_edges": len(test_edges),
              "repaired": repaired, "candidates": len(candidates),
              "prevalence": round(float(y_true.mean()), 4)},
    "heuristics": {
        "summary": summary_df.to_dict(orient="records"),
        "sampled_negatives": sampled_table.to_dict(orient="records"),
        "roc": roc_curves,
    },
    "ml": {"table": ml_table.to_dict(orient="records"),
           "feature_importance": FEATURE_IMPORTANCE},
    "temporal": temporal,
    "sna": sna,
    "communities": communities_df.to_dict(orient="records"),
    "homophily": homophily,
    "top20": top20_out.to_dict(orient="records"),
    "recommend_examples": {
        s: recommend(s, 5) for s in ["Python", "SQL", "Nursing", "Microsoft Excel"] if s in G
    },
    "best_method": best_method,
    "runtime_seconds": None,
}

def _jsonable(obj):
    if hasattr(obj, "item"):          # numpy scalars
        return obj.item()
    return str(obj)


with open(PROC / "results.json", "w", encoding="utf-8") as fh:
    json.dump(results, fh, indent=1, default=_jsonable)

print("\nartifacts written to", PROC, "|", FIG, "|", GRAPH_DIR, "|", VIZ)
print("done")
