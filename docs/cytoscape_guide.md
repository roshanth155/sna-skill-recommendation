# Cytoscape and Gephi styling guide (D3)

The graph files in `graph/` are already attributed, so nothing below needs data editing.

| File | Use |
|---|---|
| `skill_network.graphml` | Cytoscape import (attributes typed as node and edge columns) |
| `skill_network.gexf` | Gephi import |
| `skill_edges_cytoscape.csv`, `skill_nodes_cytoscape.csv` | manual import into either tool, or a spreadsheet check |
| `viz/top20_predicted_links.html` | browser view, no tools needed |

## Node attributes

| Column | Meaning |
|---|---|
| `domain` | curated domain (19 values) |
| `degree` | number of distinct skills co-demanded with this skill |
| `weighted_degree` | sum of co-occurrence weights |
| `community` | Louvain community id (0-6) |
| `betweenness` | betweenness centrality |
| `constraint` | Burt's constraint (low = structural hole) |
| `pagerank` | weighted PageRank |
| `hits_hub`, `hits_authority` | HITS scores |
| `kcore` | k-core shell |
| `generic` | true for generic/soft skills |

## Edge attributes

| Column | Meaning |
|---|---|
| `weight` | co-occurring postings; 0 for predicted edges |
| `pmi` | pointwise mutual information |
| `predicted` | 0 = observed, 1 = predicted by the model |
| `score` | Resource Allocation score (predicted edges only) |

## Cytoscape 3.10 walkthrough

1. **Import.** `File > Import > Network from File`, choose `graph/skill_network.graphml`.
   Cytoscape reads the attribute types (numeric for degree, betweenness, weight; string
   for domain). If a numeric column arrives as a string, use the `Change column type`
   control in the Table panel.
2. **Node fill by community.** In the Style panel, set *Fill Color* to a **Discrete
   Mapping** on `community`. Suggested palette (matching `figures/fig1`):
   C0 `#4E79A7`, C1 `#F28E2B`, C2 `#59A14F`, C3 `#B07AA1`, C4 `#E15759`,
   C5 `#76B7B2`, C6 `#EDC948`.
3. **Node size by degree.** *Size* to a **Continuous Mapping** on `degree`, minimum
   18, maximum 70, with a mapping curve of "Mapping2" (log-like) so hubs do not swamp
   the view.
4. **Node label.** *Label* to `skill`, font size 9, label colour `#12263A`. Turn on
   `Label Background Color` white at 70 percent opacity so labels stay readable.
5. **Edge width by weight.** *Width* to a **Continuous Mapping** on `weight`,
   minimum 0.5, maximum 5.
6. **Edge colour by type.** *Stroke Color* to a **Discrete Mapping** on `predicted`:
   `0` = `#B8C2CC`, `1` = `#C8553D`. Then *Stroke Paint* to `predicted` with
   `1` = **Dashed**, and *Stroke Width* for `1` = 3.5 so the predicted links stand out.
7. **Layout.** `Layout > Prefuse Force Directed Layout`, set "Edge weight" to
   `weight`, then `Layout > Clear All Edge Bends`. For a slide-ready figure, follow
   with `Layout > Group Attributes Layout` on `community`, which reproduces the
   clustered arrangement in the report.
8. **Focus on predictions.** Select the predicted edges with the column filter
   `predicted = 1`, then right-click `Select > Nodes > Adjacent nodes` and
   `File > New > Network > From selected nodes, all edges` to get a clean subnetwork
   of the Top-20 with their neighbourhoods.
9. **Export.** `File > Export as > Network to Image`, PNG at 600 dpi (or SVG for print).
   Name it `fig1_network_communities_cytoscape.png` so it does not overwrite the
   generated figure.

## Gephi quick path

1. `File > Open` the GEXF, accept the attribute types.
2. `Window > Layout > ForceAtlas2`, set "Edge Weight" influence to 1.2, run 2000 iterations.
3. `Appearance > Nodes > Size`: ranking on `degree`, 10 to 60.
4. `Appearance > Nodes > Colour`: partition on `community`.
5. `Appearance > Edges > Colour`: partition on `predicted`.
6. `Overview > Filters > Edge > Attribute > Equal > predicted = 1` to isolate predictions.
7. `Preview`, enable edge arrows off and labels from `skill`, then `Export > SVG/PDF`.

## Common problems

* **"The node names are missing."** The `skill` column is the node id; if your GraphML
  reader uses ids instead of labels, add a *Label* mapping from `id` or from `skill`
  in the Style panel.
* **The predicted edges have weight 0.** This is intentional: they are not observed
  co-occurrences, so they must not contribute to a weight-based layout or to
  weighted centrality. Filter them out before computing any further weighted metric.
* **Some nodes look disconnected.** Eight skills were isolated after the edge filter
  and are absent from the graph; the node table in `data/processed/nodes.csv` lists
  only the giant component that the analysis uses.
* **A pair appears twice.** Predicted edges are added only for pairs that are not
  edges of the observed graph, so a duplicate means two files were imported into one
  network. Start a fresh session.
