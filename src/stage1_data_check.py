"""
Stage 1 - Dataset load, inspection and cleaning
Project: Skill Recommendation via Link Prediction (SNA / AAT)

Runs top to bottom. Also convertible to a notebook (cells are marked with '# %%').
The code never assumes column names: it detects them and prints them, so you can
verify against your actual download.
"""

# %% [markdown]
# # Stage 1 - Dataset check and cleaning
#
# **Goal:** load the Kaggle LinkedIn job-skills table, look at what is really in it,
# normalise the skill strings, and produce one clean long table:
#
#     job_id | skill
#
# **Why a long table?** Every later step (co-occurrence graph, 80/20 split, link
# prediction) only needs "which skills appeared in which posting". Getting that
# table right and documented now prevents silent bugs later.
#
# **What this stage does NOT do:** it does not build the network yet. We first need
# to see the real skill vocabulary, because the canonical skill list (200-400 skills)
# has to be built from *your* data, not guessed.
#
# **Colab:** run `import os; os.makedirs("data/raw", exist_ok=True)` once, then
# upload `job_skills.csv` and `linkedin_job_postings.csv` into `data/raw` with the
# Files panel. Locally, just put them in `data/raw/` (see README.md).

# %%
# ---------------------------------------------------------------- 0. Setup
import os
import re
import sys
import json
from pathlib import Path

import pandas as pd

SEED = 42

# Paths. Override with environment variables if your folders differ.
DATA_DIR = Path(os.environ.get("SNA_DATA_DIR", "data/raw"))
OUT_DIR = Path(os.environ.get("SNA_OUT_DIR", "data/processed"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# A skill must appear in at least this many postings to enter the network.
# Why: pairs of rare skills co-occur once or twice, so their "edges" are noise,
# and link prediction metrics (Jaccard, Adamic-Adar) are unstable on tiny degrees.
# We revisit this number in Stage 2 once we see the real degree distribution.
MIN_POSTINGS = int(os.environ.get("SNA_MIN_POSTINGS", 200))

print("pandas", pd.__version__)
print("data dir :", DATA_DIR.resolve())
print("out dir  :", OUT_DIR.resolve())

# %%
# ------------------------------------------------- 1. Candidate input files
# The primary dataset ships as:
#   job_skills.csv                 -> one row per (posting, skill phrase)
#   linkedin_job_postings.csv      -> posting metadata (dates, title, company)
# The script tolerates other names, so you can also point it at a fallback download.
def find_file(candidates):
    for name in candidates:
        p = DATA_DIR / name
        if p.exists():
            return p
    return None

SKILLS_CSV = find_file([
    "job_skills.csv",
    "jobs/job_skills.csv",
    "linkedin_job_skills.csv",
])
POSTINGS_CSV = find_file([
    "linkedin_job_postings.csv",
    "postings.csv",
    "job_postings.csv",
])

print("skills file  :", SKILLS_CSV)
print("postings file:", POSTINGS_CSV)

if SKILLS_CSV is None:
    raise SystemExit(
        "No skills file found in "
        f"{DATA_DIR.resolve()}\n"
        "Download job_skills.csv from the Kaggle dataset and put it in that folder "
        "(see README.md for the exact files)."
    )

# %%
# ------------------------------------------------------- 2. Inspect the skills table
# dtype=str keeps IDs as text (a numeric ID would lose leading zeros and can turn
# into 1.0e+08 floats). We print everything first and check it before transforming.
skills_raw = pd.read_csv(SKILLS_CSV, dtype=str)

print("\n=== job_skills.csv ===")
print("shape      :", skills_raw.shape)
print("columns    :", list(skills_raw.columns))
print("dtypes     :\n", skills_raw.dtypes)
print("nulls      :\n", skills_raw.isna().sum())
print("duplicate rows:", int(skills_raw.duplicated().sum()))
print("\nhead:")
print(skills_raw.head(10).to_string())

# %%
# Pick the skill-text column and the posting-id column by name, and show what we picked.
def pick_column(df, must_contain, avoid=()):
    """Return the first column whose lowercase name contains one of must_contain."""
    hits = []
    for col in df.columns:
        low = col.lower()
        if any(bad in low for bad in avoid):
            continue
        if any(key in low for key in must_contain):
            hits.append(col)
    return hits[0] if hits else None

SKILL_COL = pick_column(skills_raw, ["skill"])
JOB_COL = pick_column(skills_raw, ["job_link", "job_id", "id"], avoid=["skill"])

print("SKILL_COL =", SKILL_COL)
print("JOB_COL   =", JOB_COL)
if SKILL_COL is None or JOB_COL is None:
    raise SystemExit("Could not identify the columns - paste the columns above and I will adapt.")

# %%
# Are skills comma-separated inside one cell, or one skill per row?
# We detect it instead of assuming, because both Kaggle variants exist.
sample = skills_raw[SKILL_COL].dropna().astype(str).head(5000)
sep_counts = {sep: int(sample.str.count(re.escape(sep)).sum()) for sep in [",", "|", ";"]}
SEP = max(sep_counts, key=sep_counts.get)
if sep_counts[SEP] == 0:
    SEP = None

print("separator counts:", sep_counts, "-> using:", repr(SEP))
print("\n12 sample skill cells:")
for v in sample.head(12):
    print("  ", v[:160])

# %%
# --------------------------------------------------------- 3. Raw vocabulary audit
# Why: we need to see the real strings before writing the normalisation rules.
# Counting first turns a vague "clean the data" step into a measured one.
long_raw = skills_raw[[JOB_COL, SKILL_COL]].dropna().copy()
long_raw.columns = ["job_id", "skill_raw"]

if SEP is not None:
    long_raw["skill_raw"] = long_raw["skill_raw"].str.split(SEP)
else:
    long_raw["skill_raw"] = long_raw["skill_raw"].str.split(r"\s*[,|;]\s*")

long_raw = long_raw.explode("skill_raw")
long_raw["skill_raw"] = long_raw["skill_raw"].astype(str).str.strip()
long_raw = long_raw[long_raw["skill_raw"].str.len() > 0]

raw_counts = long_raw["skill_raw"].str.lower().value_counts()
n_mentions = int(len(long_raw))
print("\nskill mentions      :", n_mentions)
print("distinct raw strings:", len(raw_counts))
print("\ntop 30 raw strings:")
print(raw_counts.head(30).to_string())

# Coverage curve: how much of the data a curated vocabulary of size N would keep.
# This is the number that justifies "we keep the top ~300 skills".
print("\ncumulative coverage of mentions by vocabulary size:")
cum = raw_counts.cumsum() / n_mentions
for n in [10, 25, 50, 100, 200, 300, 500, 1000, 2000, 5000]:
    if n <= len(raw_counts):
        print(f"  top {n:>5} raw strings -> {cum.iloc[n-1]*100:6.2f}% of all mentions")

# %%
# ------------------------------------------------------------ 4. Normalisation
# Three layers, cheapest first:
#   (a) mechanical cleanup: case, whitespace, parentheses, punctuation
#   (b) alias dictionary: known synonyms of the same skill (js -> javascript)
#   (c) optional vocabulary whitelist: drop everything outside the curated list
#
# Why a dictionary and not fuzzy matching: fuzzy matching silently merges "java"
# with "javascript" and "c" with "c++", which would corrupt the network. A checked
# dictionary is auditable and defensible in a viva.

# Characters we KEEP: + # . / - because they are part of real skill names
# (c++, c#, .net, node.js, ci/cd, scikit-learn).

ALIASES = {
    # languages
    "js": "javascript", "java script": "javascript", "javascript es6": "javascript",
    "ts": "typescript", "py": "python", "python3": "python", "python 3": "python",
    "golang": "go", "c sharp": "c#", "csharp": "c#", "cpp": "c++",
    "objective c": "objective-c",
    # frontend / backend
    "reactjs": "react", "react js": "react", "react.js": "react",
    "vuejs": "vue.js", "vue js": "vue.js", "vue": "vue.js",
    "angularjs": "angular", "angular js": "angular",
    "nodejs": "node.js", "node js": "node.js", "node": "node.js",
    "nextjs": "next.js", "next js": "next.js", "nuxtjs": "nuxt.js",
    "asp.net": "asp.net", "dotnet": ".net", "dot net": ".net",
    # databases
    "postgres": "postgresql", "postgre sql": "postgresql", "psql": "postgresql",
    "ms sql": "sql server", "mssql": "sql server", "microsoft sql server": "sql server",
    "mysql server": "mysql", "mongo db": "mongodb", "mongodb atlas": "mongodb",
    "elastic search": "elasticsearch", "dynamo db": "dynamodb",
    # cloud / devops
    "aws": "amazon web services", "amazon aws": "amazon web services",
    "gcp": "google cloud platform", "google cloud": "google cloud platform",
    "azure": "microsoft azure", "ms azure": "microsoft azure",
    "k8s": "kubernetes", "ci cd": "ci/cd", "cicd": "ci/cd",
    "continuous integration": "ci/cd", "continuous delivery": "ci/cd",
    "terraform": "terraform", "infrastructure as code": "infrastructure as code",
    # data / ml
    "sklearn": "scikit-learn", "scikit learn": "scikit-learn", "sci-kit learn": "scikit-learn",
    "tf": "tensorflow", "tensor flow": "tensorflow", "pytorch": "pytorch", "torch": "pytorch",
    "ml": "machine learning", "machine-learning": "machine learning",
    "dl": "deep learning", "ai": "artificial intelligence",
    "nlp": "natural language processing", "cv": "computer vision",
    "powerbi": "power bi", "power-bi": "power bi", "tableau": "tableau",
    "excel": "microsoft excel", "ms excel": "microsoft excel",
    "advanced excel": "microsoft excel",
    "big data": "big data", "spark": "apache spark", "pyspark": "apache spark",
    "hadoop": "apache hadoop", "airflow": "apache airflow", "kafka": "apache kafka",
    # collaboration / office
    "ms office": "microsoft office", "office 365": "microsoft office",
    "powerpoint": "microsoft powerpoint", "ms powerpoint": "microsoft powerpoint",
    "word": "microsoft word", "ms word": "microsoft word",
    "jira": "jira", "git": "git", "github": "github", "gitlab": "gitlab",
}

# Words that add nothing to a skill name when they trail it.
NOISE_SUFFIXES = (
    " skills", " skill", " programming", " programming language",
    " development", " knowledge", " experience", " proficiency", " expertise",
    " technologies", " technology", " framework", " frameworks", " tools",
)

NOISE_PREFIXES = ("proficient in ", "experience with ", "experience in ", "knowledge of ")


def normalise_skill(text: str) -> str:
    """Turn one raw skill string into a canonical lowercase form (may return '')."""
    if not isinstance(text, str):
        return ""
    s = text.lower().strip()

    # (a) mechanical cleanup
    s = s.replace("\u2013", "-").replace("\u2014", "-").replace("\u00a0", " ")
    s = re.sub(r"\([^)]*\)", " ", s)      # drop parentheticals: "sql (advanced)" -> "sql"
    s = re.sub(r"[,;|]", " ", s)          # leftover separators
    s = re.sub(r"[\"'`]", "", s)
    # strip only trailing junk; '+' and '#' are kept so that c++, c# and
    # objective-c survive, and a leading '.' is kept so that .net survives
    s = re.sub(r"\s+", " ", s).strip()
    s = s.rstrip(" .*_").lstrip()
    if not s:
        return ""

    for pref in NOISE_PREFIXES:
        if s.startswith(pref):
            s = s[len(pref):]
    for suf in NOISE_SUFFIXES:
        if s.endswith(suf) and len(s) > len(suf) + 1:
            s = s[: -len(suf)]
    s = s.strip()

    # (b) alias layer
    s = ALIASES.get(s, s)
    return s


# Sanity check the normaliser on known-tricky inputs.
tests = ["JS", "React.js", "Postgres", "C++", "C#", ".NET", "Node JS",
         "Python Programming", "Proficient in SQL", "Power BI", "scikit learn"]
for t in tests:
    print(f"  {t!r:28} -> {normalise_skill(t)!r}")

# %%
# Apply normalisation to the full raw table.
long_raw["skill"] = long_raw["skill_raw"].map(normalise_skill)
long_raw = long_raw[long_raw["skill"].str.len() > 0]

# Deduplicate WITHIN a posting: a posting listing "python, Python, python3"
# must count once, otherwise co-occurrence weights are inflated.
long = long_raw.drop_duplicates(["job_id", "skill"])[["job_id", "skill"]].copy()

print("\npostings (unique job ids):", long["job_id"].nunique())
print("skill mentions after dedup:", len(long))
print("canonical skills         :", long["skill"].nunique())

canon_counts = long["skill"].value_counts()
print("\ntop 40 canonical skills:")
print(canon_counts.head(40).to_string())

# %%
# ------------------------------------------- 5. Optional curated whitelist hook
# Leave as None for now. In Stage 1b we paste a reviewed 200-400 skill vocabulary
# here; anything outside it is dropped. This is what stops "ability to multitask"
# style phrases from becoming network nodes.
VOCAB = None

if VOCAB is not None:
    before = len(long)
    long = long[long["skill"].isin(VOCAB)]
    print(f"whitelist applied: kept {len(long):,}/{before:,} mentions")

# %%
# --------------------------------------------- 6. Drop rare skills, report losses
counts = long["skill"].value_counts()
kept_skills = counts[counts >= MIN_POSTINGS].index
long_clean = long[long["skill"].isin(kept_skills)].copy()

print(f"\nMIN_POSTINGS = {MIN_POSTINGS}")
if len(kept_skills) == 0:
    print("  WARNING: no skill reached MIN_POSTINGS - lower the threshold "
          "(SNA_MIN_POSTINGS=50) or check that the skills column was parsed correctly.")
print(f"skills kept     : {len(kept_skills):,} of {long['skill'].nunique():,}")
print(f"mentions kept   : {len(long_clean)/max(len(long),1)*100:6.2f}%")
print(f"postings kept   : {long_clean['job_id'].nunique():,} of {long['job_id'].nunique():,}")

per_posting = long_clean.groupby("job_id")["skill"].nunique()
print("\nskills per posting after cleaning:")
if len(per_posting):
    print(per_posting.describe().to_string())
    print("\npostings by number of skills (1-15+):")
    print(per_posting.clip(upper=15).value_counts().sort_index().to_string())
else:
    print("  (empty)")

# %%
# ------------------------------------------------- 7. Postings metadata (optional)
# Needed later for the temporal validation split (older postings -> newer postings),
# so we check the date columns now instead of discovering a problem in Stage 3.
# The postings file can be large, so we inspect the header first and then load
# only the handful of columns we actually need (id + dates + title).
postings = None
if POSTINGS_CSV is not None:
    header = pd.read_csv(POSTINGS_CSV, nrows=5, dtype=str)
    all_cols = list(header.columns)
    print("\n=== postings file ===")
    print("columns:", all_cols)
    print(header.head(3).to_string())

    id_like = [c for c in all_cols if c.lower() in ("job_link", "job_id", "id")]
    date_like = [c for c in all_cols
                 if any(k in c.lower() for k in ["date", "time", "listed", "seen", "posted"])]
    title_like = [c for c in all_cols if "title" in c.lower()]
    usecols = list(dict.fromkeys(id_like[:2] + date_like[:6] + title_like[:1]))
    print("date-like columns:", date_like)
    print("loading only    :", usecols)

    postings = pd.read_csv(POSTINGS_CSV, dtype=str, usecols=usecols)
    print("shape loaded    :", postings.shape)

    # Join rate: if most postings have no skills row, the network is built on a
    # biased sub-sample and we must say so in the report.
    key = next((c for c in postings.columns if c.lower() in ("job_link", "job_id", "id")), None)
    if key is not None and len(long_clean):
        overlap = long_clean["job_id"].isin(set(postings[key].dropna())).mean()
        print(f"join key '{key}': {overlap*100:.2f}% of cleaned postings found in postings file")

    if date_like:
        dt = pd.to_datetime(postings[date_like[0]], errors="coerce", utc=True)
        print(f"{date_like[0]}: {dt.notna().mean()*100:.1f}% parsed, "
              f"range {dt.min()} .. {dt.max()}")
    else:
        print("No parseable date column -> temporal split unavailable, use a grouped split.")
else:
    print("\nNo postings file found - temporal split will be replaced by a grouped split.")

# %%
# ------------------------------------------------------------------ 8. Save
long_path = OUT_DIR / "skills_long.csv.gz"
audit_path = OUT_DIR / "stage1_audit.txt"

long_clean.to_csv(long_path, index=False, compression="gzip")

report_lines = [
    "STAGE 1 AUDIT",
    f"skills file          : {SKILLS_CSV.name}",
    f"postings file        : {POSTINGS_CSV.name if POSTINGS_CSV else 'none'}",
    f"skills table shape   : {skills_raw.shape}",
    f"skills table columns : {list(skills_raw.columns)}",
    f"skill column used    : {SKILL_COL}",
    f"job id column used   : {JOB_COL}",
    f"separator detected   : {SEP!r}",
    f"raw distinct strings : {len(raw_counts):,}",
    f"canonical skills     : {long['skill'].nunique():,}",
    f"MIN_POSTINGS         : {MIN_POSTINGS}",
    f"skills kept          : {len(kept_skills):,}",
    f"postings with skills : {long_clean['job_id'].nunique():,}",
    f"rows in long table   : {len(long_clean):,}",
    "",
    "TOP 60 CANONICAL SKILLS (postings count):",
    canon_counts.head(60).to_string(),
]
audit_path.write_text("\n".join(report_lines), encoding="utf-8")

print("\nsaved:", long_path, f"({long_path.stat().st_size/1e6:.1f} MB)")
print("saved:", audit_path)

# %%
# --------------------------------------------------- 9. What to send back
# Copy the whole printed output (or stage1_audit.txt) so the next stage can be
# built on the real vocabulary instead of assumptions.
print("\n" + "=" * 70)
print("PASTE BACK: the printed report (columns, shapes, top-60 canonical skills,")
print("plus 20 sample rows of the long table).")
print("=" * 70)
if len(long_clean):
    print(long_clean.sample(min(20, len(long_clean)), random_state=SEED).to_string(index=False))
else:
    print("(no rows survived cleaning - see the WARNING above)")
