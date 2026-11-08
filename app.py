import json
import math
from itertools import combinations
from pathlib import Path

import networkx as nx
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

ROOT = Path(__file__).resolve().parent
GRAPH = ROOT / "graph" / "skill_network.graphml"
VIZ = ROOT / "viz" / "top20_predicted_links.html"
RESULTS = ROOT / "data" / "processed" / "results.json"
FIGS = ROOT / "figures"

METHODS = {
    "Adamic-Adar": "adamic_adar",
    "Resource Allocation": "resource_allocation",
    "Common Neighbours": "shared",
    "Jaccard": "jaccard",
}
GENERIC = ["Communication", "Teamwork", "Problem Solving", "Leadership"]

st.set_page_config(page_title="Skill Recommender", layout="wide")


@st.cache_resource
def load_graph():
    raw = nx.read_graphml(GRAPH)
    names = {
        n: str(d.get("label") or d.get("name") or d.get("skill") or n)
        for n, d in raw.nodes(data=True)
    }
    G = nx.Graph(nx.relabel_nodes(raw, names))
    if all("community" in d for _, d in G.nodes(data=True)):
        comm = {n: d["community"] for n, d in G.nodes(data=True)}
    else:
        comm = {}
        for i, c in enumerate(nx.community.louvain_communities(G, seed=42)):
            for n in c:
                comm[n] = i
    return G, comm


@st.cache_data
def score_all_pairs():
    G, comm = load_graph()
    rows = []
    for u, v in combinations(G.nodes, 2):
        if G.has_edge(u, v):
            continue
        cn = list(nx.common_neighbors(G, u, v))
        if not cn:
            continue
        union = len(set(G[u]) | set(G[v]))
        rows.append(
            {
                "a": u,
                "b": v,
                "shared": len(cn),
                "jaccard": len(cn) / union,
                "adamic_adar": sum(1 / math.log(G.degree(z)) for z in cn if G.degree(z) > 1),
                "resource_allocation": sum(1 / G.degree(z) for z in cn),
                "deg_a": G.degree(u),
                "deg_b": G.degree(v),
                "cross_community": comm[u] != comm[v],
            }
        )
    return pd.DataFrame(rows)


def via(G, a, b, n=3):
    cn = sorted(nx.common_neighbors(G, a, b), key=G.degree, reverse=True)
    return ", ".join(cn[:n])


st.title("Skill recommender")
st.caption("Which skills do job postings demand together? Predicted from a skill co-occurrence network.")

if not GRAPH.exists():
    st.error(
        "graph/skill_network.graphml not found. Run `python src/pipeline.py` first "
        "and launch this app from the repo root."
    )
    st.stop()

G, comm = load_graph()
pairs = score_all_pairs()

tab_rec, tab_top, tab_net, tab_res = st.tabs(
    ["Recommend", "Top predicted pairs", "Network", "Results"]
)

with tab_rec:
    c1, c2, c3 = st.columns([3, 1, 2])
    skill = c1.selectbox("Your skill", sorted(G.nodes), index=None, placeholder="Type or pick a skill")
    k = c2.number_input("How many", 1, 30, 10)
    method = c3.selectbox("Ranking method", list(METHODS))
    if skill:
        col = METHODS[method]
        sub = pairs[(pairs.a == skill) | (pairs.b == skill)].copy()
        sub["Skill"] = sub.apply(lambda r: r.b if r.a == skill else r.a, axis=1)
        sub = sub.sort_values([col, "resource_allocation", "Skill"], ascending=[False, False, True]).head(int(k))
        if sub.empty:
            st.info("No unconnected skills share neighbours with this one.")
        else:
            out = pd.DataFrame(
                {
                    "Skill": sub["Skill"],
                    "Score": sub[col].round(3),
                    "Shared neighbours": sub["shared"],
                    "Strongest links in common": [via(G, skill, s) for s in sub["Skill"]],
                    "Crosses communities": sub["cross_community"].map({True: "yes", False: "no"}),
                }
            )
            st.dataframe(out, hide_index=True, use_container_width=True)
            st.caption(
                f"{skill} already pairs with {G.degree(skill)} skills. "
                "These are skills it is not yet linked to."
            )

with tab_top:
    PRED = ROOT / "data" / "processed" / "predictions_top20.csv"
    NODES = ROOT / "data" / "processed" / "nodes.csv"
    if PRED.exists():
        p = pd.read_csv(PRED)
        st.subheader("Top 20 predicted skill pairs")
        st.caption("Straight from the pipeline, filtered and ranked as in the report.")
        same = p.same_community.astype(str).str.lower().map({"true": "yes", "false": "no"})
        st.dataframe(
            pd.DataFrame(
                {
                    "Skill A": p.skill_a,
                    "Skill B": p.skill_b,
                    "Score": p.score.round(3),
                    "Shared neighbours": p.cn,
                    "Same community": same,
                    "Domains": p.domain_a + " / " + p.domain_b,
                    "Common neighbours include": p.shared_neighbours,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
    else:
        st.info("data/processed/predictions_top20.csv not found. Run the pipeline first.")

    generic_default = []
    if NODES.exists():
        nd = pd.read_csv(NODES)
        flag = nd["generic"].astype(str).str.lower() == "true"
        generic_default = [s for s in nd.loc[flag, "skill"] if s in G.nodes]

    with st.expander("Explore with your own filters"):
        f1, f2, f3 = st.columns(3)
        min_shared = f1.slider("Minimum shared neighbours", 1, 15, 5)
        min_deg = f2.slider("Minimum degree of each skill", 1, 15, 5)
        cap = f3.checkbox("Show each skill at most twice", value=True)
        generic = st.multiselect("Leave out generic skills", sorted(G.nodes), default=generic_default)
        t = pairs[
            (pairs.shared >= min_shared)
            & (pairs.deg_a >= min_deg)
            & (pairs.deg_b >= min_deg)
            & ~pairs.a.isin(generic)
            & ~pairs.b.isin(generic)
        ].sort_values("adamic_adar", ascending=False)
        if cap:
            seen, keep = {}, []
            for i, r in t.iterrows():
                if seen.get(r.a, 0) < 2 and seen.get(r.b, 0) < 2:
                    keep.append(i)
                    seen[r.a] = seen.get(r.a, 0) + 1
                    seen[r.b] = seen.get(r.b, 0) + 1
            t = t.loc[keep]
        t = t.head(20)
        st.dataframe(
            pd.DataFrame(
                {
                    "Skill A": t.a,
                    "Skill B": t.b,
                    "Adamic-Adar": t.adamic_adar.round(3),
                    "Shared neighbours": t.shared,
                }
            ),
            hide_index=True,
            use_container_width=True,
        )
        st.caption("These scores are recomputed in the app, so they use a different scale from the table above.")

with tab_net:
    if VIZ.exists():
        components.html(VIZ.read_text(encoding="utf-8"), height=900, scrolling=True)
    else:
        st.info("viz/top20_predicted_links.html not found. Run the pipeline to create it.")

with tab_res:
    if RESULTS.exists():
        with st.expander("Full results.json"):
            st.json(json.loads(RESULTS.read_text()))
    figs = sorted(FIGS.glob("*.png")) if FIGS.exists() else []
    for i in range(0, len(figs), 2):
        cols = st.columns(2)
        for col, f in zip(cols, figs[i : i + 2]):
            col.image(str(f), caption=f.stem, use_container_width=True)
    if not RESULTS.exists() and not figs:
        st.info("No results or figures found yet. Run the pipeline first.")
