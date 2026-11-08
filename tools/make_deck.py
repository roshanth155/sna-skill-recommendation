"""Build the presentation deck (D5) and its Markdown outline.

Both artefacts come from the same SLIDES list, so the slides and the outline can
never disagree. Numbers are read from data/processed/results.json.

Usage:  python tools/make_deck.py
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image as PILImage
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Inches, Pt

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

R = json.loads((ROOT / "data" / "processed" / "results.json").read_text(encoding="utf-8"))
S = {r["method"]: r for r in R["heuristics"]["summary"]}
SAMPLED = {r["method"]: r for r in R["heuristics"]["sampled_negatives"]}
ML = {r["model"]: r for r in R["ml"]["table"]}
SNA, HOM, TMP = R["sna"], R["homophily"], R["temporal"]
TOP20 = R["top20"]

NAVY = RGBColor(0x12, 0x26, 0x3A)
ACCENT = RGBColor(0xC8, 0x55, 0x3D)
GREY = RGBColor(0x5A, 0x6B, 0x7B)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1B, 0x1B, 0x1B)
PALE = RGBColor(0xF2, 0xF5, 0xF8)

W, H = 13.333, 7.5

SLIDES = [
    {
        "kind": "title",
        "title": "Which skills does the market demand together?",
        "subtitle": "Link prediction on a skill co-occurrence network built from 27,846 job postings",
        "footer": "Social Network Analysis (AAT)  |  Course project  |  November 2026",
        "notes": "Open with the question, not the method: choosing a skill is a bet on a bundle. "
                 "One sentence on the data (LinkedIn postings, Aug-Nov 2023) and one on the output "
                 "(Top-20 skill pairs plus a learn-next function). Keep this slide under 30 seconds.",
    },
    {
        "kind": "bullets",
        "kicker": "Problem",
        "title": "A job posting is a bundle, not a list",
        "bullets": [
            (0, "Python, SQL and AWS appear in one advert because one role needs all three"),
            (1, "So the value of a skill depends on what it is used with"),
            (0, "Observed bundles give a network: skills are nodes, co-demand is an edge"),
            (0, "The interesting structure is what is missing, not what is popular"),
            (1, "Link prediction: which unobserved pairs are likely to be demanded together?"),
            (0, "Output: Top-20 pairs the market will likely demand together, and learn-next(skill)"),
        ],
        "notes": "Motivate the shift from single-skill popularity to pairs. The last bullet is the "
                 "product; everything after this slide is the evidence behind it. Mention that the "
                 "same machinery answers a second question: what to learn next given skill X.",
    },
    {
        "kind": "bullets",
        "kicker": "Modelling",
        "title": "From postings to a skill network",
        "bullets": [
            (0, "Bipartite incidence: postings x skills, 27,846 x 165 after cleaning"),
            (0, "Project onto skills: edge weight = postings mentioning both skills"),
            (0, "Undirected because co-occurrence is symmetric; direction was never in the data"),
            (0, "Two edge conditions: support (weight >= 50) and selectivity (PMI >= 0.5)"),
            (0, "Keep the giant component: 157 nodes, 883 edges, density 0.072"),
            (0, "Eight isolated skills reported, not silently dropped"),
        ],
        "notes": "Explain the projection carefully; this is the modelling decision examiners probe. "
                 "Undirected is not a simplification: a posting gives identical evidence both ways. "
                 "The two-condition edge rule is the next slide's topic.",
    },
    {
        "kind": "bullets",
        "kicker": "Data",
        "title": "33,246 postings, three months, one vocabulary",
        "bullets": [
            (0, "Source: LinkedIn postings (arshkon schema), Hugging Face mirror, Aug 10 - Nov 4 2023"),
            (1, "Kaggle 1.3M skills table also supported: pipeline has a skills_table load path"),
            (0, "245 curated skills, 19 domains, alias table, no fuzzy matching"),
            (1, "Fuzzy matching would merge java with javascript and c with c++"),
            (0, "27,846 postings keep at least two recognised skills (83.8 percent)"),
            (0, "165 skills above the rarity filter of 100 postings; 155,027 mentions"),
            (0, "Mean 5.57 skills per posting; deduplicated within each posting"),
        ],
        "notes": "Be explicit that this is a mirror and a three-month window, not the full Kaggle "
                 "corpus. That is a scope limit, stated up front. The vocabulary decision matters: "
                 "precision over recall, because a noisy node set corrupts every downstream number.",
    },
    {
        "kind": "bullets_image",
        "kicker": "Network",
        "title": "Why an edge needs support and selectivity",
        "bullets": [
            (0, "Count alone keeps chance pairs: Excel and Leadership meet in any threshold"),
            (0, "PMI compares observed co-occurrence with what popularity predicts"),
            (0, "Chosen: weight >= 50 and PMI >= 0.5, picked from a 15-point sensitivity grid"),
            (1, "157 of 165 skills stay connected; moving to t = 100 leaves 34 components"),
            (0, "Result: 883 edges, mean degree 11.25, max degree 53 (Python)"),
        ],
        "image": "figures/fig3_edge_weight_threshold.png",
        "notes": "This is the methodological decision worth the most marks. A count threshold keeps "
                 "popularity pairs; PMI removes them. Point at the red line on the histogram and say "
                 "what would happen on either side: too low and the graph is dense and trivial, too "
                 "high and it shatters into dozens of components.",
    },
    {
        "kind": "bullets",
        "kicker": "Validation design",
        "title": "Four guards against fooling ourselves",
        "bullets": [
            (0, "Hide a random 20 percent of edges: 710 train, 173 test, 19.6 percent"),
            (1, "Four edges repaired for connectivity and excluded from the test set"),
            (0, "Score all 11,536 non-edges: exact Precision@K, no sampling noise"),
            (0, "Repeat over 25 splits and report mean with standard deviation"),
            (0, "Two negative regimes: uniform random non-edges and distance-2 hard negatives"),
            (0, "Temporal split: train before 2023-11-02, test on pairs seen only afterwards"),
        ],
        "notes": "The exam question is always leakage. Walk through the three answers: features come "
                 "from the training graph only, the classifier's fitting half is disjoint from its "
                 "scoring half, and the temporal split removes the possibility that a hidden edge is "
                 "recoverable from elsewhere in the same corpus.",
    },
    {
        "kind": "bullets",
        "kicker": "Indices",
        "title": "Five indices, implemented by hand",
        "bullets": [
            (0, "Common Neighbours: |N(u) intersect N(v)|"),
            (0, "Jaccard: intersection / union - penalises large neighbourhoods"),
            (0, "Adamic-Adar: sum 1/log(deg(w)) over shared neighbours"),
            (0, "Preferential Attachment: deg(u) * deg(v), uses no neighbourhood"),
            (0, "Resource Allocation: sum 1/deg(w), a sharper Adamic-Adar"),
            (0, "Cross-checked against NetworkX on 200 pairs: maximum difference 0"),
        ],
        "notes": "Say the formulas, then the cross-check result. The assertion in the code would fail "
                 "loudly if the hand implementation drifted from NetworkX, which is the answer to "
                 "'did you implement it or call a library'. Mention that the graph passed to the "
                 "indices is unweighted, which is the textbook definition.",
    },
    {
        "kind": "table",
        "kicker": "Results",
        "title": "Neighbourhood indices work; degree alone does not",
        "headers": ["Method", "P@10", "P@20", "P@50", "P@100", "AUC"],
        "rows": [
            ["Common Neighbours", f"{S['CN']['P@10']:.3f}", f"{S['CN']['P@20']:.3f}",
             f"{S['CN']['P@50']:.3f}", f"{S['CN']['P@100']:.3f}", f"{S['CN']['AUC']:.4f}"],
            ["Jaccard", f"{S['Jaccard']['P@10']:.3f}", f"{S['Jaccard']['P@20']:.3f}",
             f"{S['Jaccard']['P@50']:.3f}", f"{S['Jaccard']['P@100']:.3f}", f"{S['Jaccard']['AUC']:.4f}"],
            ["Adamic-Adar", f"{S['Adamic-Adar']['P@10']:.3f}", f"{S['Adamic-Adar']['P@20']:.3f}",
             f"{S['Adamic-Adar']['P@50']:.3f}", f"{S['Adamic-Adar']['P@100']:.3f}",
             f"{S['Adamic-Adar']['AUC']:.4f}"],
            ["Preferential Attachment", f"{S['Preferential Attachment']['P@10']:.3f}",
             f"{S['Preferential Attachment']['P@20']:.3f}", f"{S['Preferential Attachment']['P@50']:.3f}",
             f"{S['Preferential Attachment']['P@100']:.3f}", f"{S['Preferential Attachment']['AUC']:.4f}"],
            ["Resource Allocation", f"{S['Resource Allocation']['P@10']:.3f}",
             f"{S['Resource Allocation']['P@20']:.3f}", f"{S['Resource Allocation']['P@50']:.3f}",
             f"{S['Resource Allocation']['P@100']:.3f}", f"{S['Resource Allocation']['AUC']:.4f}"],
        ],
        "note": "Mean over 25 random splits; 11,536 candidates with 1.5 percent positives. "
                "A random ranking would score about 0.015.",
        "notes": "Lead with the base rate, because 0.15 precision sounds weak until it is compared "
                 "with 0.015. Then the ordering: AA and RA best on AUC, CN best at K=10, Jaccard "
                 "penalised twice by the union, PA worst. The PA explanation is assortativity (-0.126) "
                 "and generalist hubs, which is the syllabus link.",
    },
    {
        "kind": "bullets_image",
        "kicker": "Ranking quality",
        "title": "The ordering is right, not just the top of the list",
        "bullets": [
            (0, "Best single index: Resource Allocation, AUC 0.8837"),
            (0, "Adamic-Adar close behind at 0.8819, best Precision@20 (0.142)"),
            (0, "Standard deviation around 0.03 across splits"),
            (0, "Easy negatives inflate everything: P@20 rises to 0.98-0.995"),
            (0, "Distance-2 negatives are the honest test: AUC falls to 0.67-0.81"),
            (1, "Method ordering is stable in every regime"),
        ],
        "image": "figures/fig4_roc_curves.png",
        "notes": "Two points. First, the neighbourhood indices separate from PA. Second, and more "
                 "important, the absolute numbers depend heavily on the negative distribution, so "
                 "quote the ordering of methods rather than a single AUC. Mention the hard-negative "
                 "table here if asked.",
    },
    {
        "kind": "bullets",
        "kicker": "Comparison",
        "title": "Supervised models and a forward-looking test",
        "bullets": [
            (0, f"Random Forest AUC {ML['Random Forest']['AUC']:.4f}, the best of anything tested"),
            (1, "Adamic-Adar and Resource Allocation carry 48 percent of importance"),
            (0, f"Logistic Regression AUC {ML['Logistic Regression']['AUC']:.4f}, best P@20 (0.150)"),
            (0, "Temporal split, 100 forward-looking positives:"),
            (1, f"CN P@20 {TMP['methods']['CN']['P@20']:.2f}, AUC {TMP['methods']['CN']['AUC']:.3f}"),
            (1, f"Adamic-Adar P@20 {TMP['methods']['Adamic-Adar']['P@20']:.2f}, RA {TMP['methods']['Resource Allocation']['P@20']:.2f}"),
            (1, f"Jaccard collapses: P@20 {TMP['methods']['Jaccard']['P@20']:.2f}, AUC {TMP['methods']['Jaccard']['AUC']:.3f}"),
        ],
        "notes": "The forest buys about three points of AUC over the best heuristic but needs labelled "
                 "pairs, so the heuristics stay the defensible core. Then the temporal slide-within-a-"
                 "slide: the neighbourhood signal survives forward in time, Jaccard does not, and PA "
                 "has a decent AUC with no precision at the top - which is why both metrics are shown.",
    },
    {
        "kind": "bullets_image",
        "kicker": "SNA analysis",
        "title": "Seven communities, only partly aligned with job domains",
        "bullets": [
            (0, f"Louvain: {SNA['n_communities']} communities, modularity {SNA['modularity']}"),
            (0, f"NMI against 19 curated domains: {SNA['nmi_communities_vs_domains']}"),
            (0, "Design and healthcare are pure clusters (80-100 percent own domain)"),
            (0, "The 56-skill software cluster mixes programming, cloud, data and delivery"),
            (0, "Domains are labels; communities are what employers actually bundle"),
        ],
        "image": "figures/fig1_network_communities.png",
        "notes": "The key line is the last bullet: a low domain share inside the big communities is "
                 "not a failure of the algorithm, it is the finding. Employers bundle across domains. "
                 "Point at the labelled clusters and name two or three.",
    },
    {
        "kind": "bullets_image",
        "kicker": "SNA analysis",
        "title": "Small-world, homophilous, disassortative",
        "bullets": [
            (0, f"Clustering {SNA['avg_clustering']} against {SNA['random_clustering']} for a random graph"),
            (0, f"Average path length {SNA['avg_path_length']} steps; small-world sigma {SNA['small_world_sigma']}"),
            (0, f"Transitivity {SNA['transitivity']}: the triadic closure that makes CN work"),
            (0, f"Homophily: {HOM['observed_edges_same_community']} of observed edges inside a community, "
                f"baseline {HOM['random_non_edge_baseline']}"),
            (0, f"Degree assortativity {SNA['assortativity_degree']}: why Preferential Attachment fails"),
            (0, f"Core: k-core {SNA['max_kcore']} - Python, Agile, SQL, AWS, Azure, Java, CI/CD"),
        ],
        "image": "figures/fig2_degree_distribution.png",
        "notes": "This slide links the structure to the prediction results: clustering explains why CN "
                 "works, assortativity explains why PA does not, and homophily explains why the "
                 "recommendations are mostly within-community. The degree tail is heavy but not a clean "
                 "power law (slope -0.78, R2 0.68), which is honest given 157 nodes.",
    },
    {
        "kind": "bullets_image",
        "kicker": "Recommendations",
        "title": "Top-20 pairs: mostly bridges between communities",
        "bullets": [
            (0, "11,363 unobserved pairs, 548 survive the filters, Top-20 diversified"),
            (0, "Python + Stakeholder Management: 22 shared neighbours, cross-community"),
            (0, "Excel + Payroll, Accounting + Compliance: finance back office"),
            (0, "C# + REST API, Kubernetes + REST API, Angular + Azure: web and cloud convergence"),
            (0, f"{HOM['predicted_top20_same_community']:.0%} of predicted pairs are same-community, "
                f"against {HOM['observed_edges_same_community']:.0%} observed"),
        ],
        "image": "figures/fig6_top20_predicted_pairs.png",
        "notes": "Explain the filters first (5 shared neighbours, no generic endpoint, minimum degree 5, "
                 "diversity cap 2), because they are why the list is readable. Then the bridging point: "
                 "the model proposes more cross-community pairs than the network contains, which is "
                 "either the market moving or the model extrapolating - and say which evidence would "
                 "settle it.",
    },
    {
        "kind": "table",
        "kicker": "Recommendations",
        "title": "The list, and what to learn next",
        "headers": ["#", "Skill A", "Skill B", "Score", "CN"],
        "rows": [[str(i + 1), r["skill_a"], r["skill_b"], f"{r['score']:.3f}", str(r["cn"])]
                 for i, r in enumerate(TOP20[:10])],
        "note": "recommend('Python') -> Stakeholder Management 1.54, CSS 0.68, Angular 0.54, "
                "Project Management 0.39. recommend('Nursing') -> no pair passes the filters.",
        "notes": "Read three rows aloud and interpret them: Python to stakeholder management is the "
                 "hybrid-role story, Excel to Payroll is the finance story, C# to REST API is the "
                 "full-stack story. Then the empty Nursing answer: the filters refuse to invent advice, "
                 "which is a feature of an evidence-based recommender.",
    },
    {
        "kind": "bullets",
        "kicker": "Close",
        "title": "What this establishes, and what it does not",
        "bullets": [
            (0, "Neighbourhood link prediction works on skill co-occurrence, and survives a temporal shift"),
            (0, "The choice of index matters; the choice of negative distribution matters more"),
            (0, "Co-occurrence is not complementarity: the graph records bundles, not causation"),
            (0, "Three-month mirror sample, 165-skill vocabulary, 173 test edges per split"),
            (0, "Next: posting-level features (seniority, salary), a larger corpus, forward validation"),
            (0, "Takeaway: the useful recommendations are the cross-community ones"),
        ],
        "notes": "End on the limitation that matters most, then the takeaway, not the other way round. "
                 "If there is one sentence to leave in the examiner's head: the market's missing links "
                 "are mostly between technical and business skills, and that is what a learner should "
                 "act on.",
    },
]


def add_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def box(slide, left, top, width, height, color):
    from pptx.enum.shapes import MSO_SHAPE
    shape = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(left), Inches(top),
                                   Inches(width), Inches(height))
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()
    shape.shadow.inherit = False
    return shape


def text(slide, left, top, width, height, value, size, color=INK, bold=False,
         align=PP_ALIGN.LEFT, space_after=0):
    frame = slide.shapes.add_textbox(Inches(left), Inches(top), Inches(width), Inches(height))
    tf = frame.text_frame
    tf.word_wrap = True
    parts = value.split("\n")
    for i, part in enumerate(parts):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.text = part
        p.alignment = align
        p.space_after = Pt(space_after)
        for run in p.runs:
            run.font.size = Pt(size)
            run.font.bold = bold
            run.font.color.rgb = color
            run.font.name = "Calibri"
    return frame


def add_slide(prs, spec, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_bg(slide, WHITE)

    if spec["kind"] == "title":
        add_bg(slide, NAVY)
        box(slide, 0, 0, 0.22, H, ACCENT)
        text(slide, 0.9, 1.9, 11.4, 1.9, spec["title"], 40, WHITE, bold=True)
        text(slide, 0.9, 3.9, 10.6, 1.0, spec["subtitle"], 17, RGBColor(0xC9, 0xD6, 0xE2))
        box(slide, 0.95, 5.05, 1.6, 0.06, ACCENT)
        text(slide, 0.9, 5.3, 10.6, 0.5, spec["footer"], 12, RGBColor(0x9F, 0xB0, 0xC2))
        if spec.get("notes"):
            slide.notes_slide.notes_text_frame.text = spec["notes"]
        return slide

    box(slide, 0, 0, W, 0.09, NAVY)
    box(slide, 0.55, 0.62, 0.16, 0.16, ACCENT)
    text(slide, 0.85, 0.5, 8.0, 0.4, spec["kicker"].upper(), 11.5, ACCENT, bold=True)
    text(slide, 0.55, 0.95, 12.2, 0.9, spec["title"], 27, NAVY, bold=True)

    has_image = "image" in spec
    left_w = 6.2 if has_image else 12.2

    if "bullets" in spec:
        frame = slide.shapes.add_textbox(Inches(0.6), Inches(1.95), Inches(left_w), Inches(4.9))
        tf = frame.text_frame
        tf.word_wrap = True
        first = True
        for level, item in spec["bullets"]:
            p = tf.paragraphs[0] if first else tf.add_paragraph()
            first = False
            p.text = ("- " if level == 0 else "") + item
            p.level = level
            p.space_after = Pt(9 if level == 0 else 4)
            p.space_before = Pt(0 if level == 0 else 0)
            for run in p.runs:
                run.font.name = "Calibri"
                run.font.size = Pt(15.5 if level == 0 else 13)
                run.font.color.rgb = INK if level == 0 else GREY
                run.font.bold = False
            if level == 1:
                p.left_indent = Inches(0.3)

    if has_image:
        path = ROOT / spec["image"]
        if path.exists():
            with PILImage.open(path) as im:
                iw, ih = im.size
            max_w, max_h = 6.35, 4.55
            scale = min(max_w / iw, max_h / ih)
            w_in, h_in = iw * scale, ih * scale
            slide.shapes.add_picture(str(path), Inches(W - 0.55 - w_in),
                                     Inches(1.62 + (max_h - h_in) / 2),
                                     Inches(w_in), Inches(h_in))

    if "headers" in spec:
        n_rows = len(spec["rows"]) + 1
        n_cols = len(spec["headers"])
        top = 1.95
        height = min(4.2, 0.40 * n_rows)
        numeric = [j for j, h in enumerate(spec["headers"])
                   if h in ("P@10", "P@20", "P@50", "P@100", "AUC", "Score", "CN")]
        shape = slide.shapes.add_table(n_rows, n_cols, Inches(0.9), Inches(top),
                                       Inches(11.5), Inches(height))
        table = shape.table
        widths = [2.6] + [1.7] * (n_cols - 1)
        total = sum(widths)
        for j, wd in enumerate(widths):
            table.columns[j].width = Inches(11.5 * wd / total)
        for j, head in enumerate(spec["headers"]):
            cell = table.cell(0, j)
            cell.text = head
            for p in cell.text_frame.paragraphs:
                for run in p.runs:
                    run.font.size = Pt(12)
                    run.font.bold = True
                    run.font.color.rgb = WHITE
                    run.font.name = "Calibri"
            cell.fill.solid()
            cell.fill.fore_color.rgb = NAVY
        for i, row in enumerate(spec["rows"], start=1):
            for j, value in enumerate(row):
                cell = table.cell(i, j)
                cell.text = str(value)
                for p in cell.text_frame.paragraphs:
                    p.alignment = PP_ALIGN.RIGHT if j in numeric else PP_ALIGN.LEFT
                    for run in p.runs:
                        run.font.size = Pt(12)
                        run.font.color.rgb = INK
                        run.font.name = "Calibri"
                cell.fill.solid()
                cell.fill.fore_color.rgb = PALE if i % 2 == 0 else WHITE
        if "note" in spec:
            text(slide, 0.9, top + height + 0.18, 11.4, 0.7, spec["note"], 11, GREY)

    if spec.get("notes"):
        slide.notes_slide.notes_text_frame.text = spec["notes"]
    return slide


def write_outline() -> None:
    lines = ["# Deck outline (D5)", "",
             "Generated from the same slide specification as `docs/deck.pptx`, with speaker notes.",
             ""]
    for i, spec in enumerate(SLIDES, start=1):
        lines.append(f"## Slide {i}. {spec['title']}")
        if "subtitle" in spec:
            lines.append(f"*{spec['subtitle']}*")
        if "bullets" in spec:
            for level, item in spec["bullets"]:
                lines.append(("  " * level) + f"- {item}")
        if "headers" in spec:
            lines.append("")
            lines.append("| " + " | ".join(spec["headers"]) + " |")
            lines.append("|" + "---|" * len(spec["headers"]))
            for row in spec["rows"]:
                lines.append("| " + " | ".join(row) + " |")
        if "note" in spec:
            lines.append("")
            lines.append(f"*{spec['note']}*")
        if "image" in spec:
            lines.append("")
            lines.append(f"Figure: `{spec['image']}`")
        lines.append("")
        lines.append(f"**Speaker notes.** {spec['notes']}")
        lines.append("")
    (DOCS / "deck_outline.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote docs/deck_outline.md")


def main() -> None:
    prs = Presentation()
    prs.slide_width = Inches(W)
    prs.slide_height = Inches(H)
    for i, spec in enumerate(SLIDES, start=1):
        add_slide(prs, spec, i)
    out = DOCS / "deck.pptx"
    prs.save(out)
    print(f"wrote {out} ({len(SLIDES)} slides, {out.stat().st_size/1024:.0f} KB)")
    write_outline()


if __name__ == "__main__":
    main()
