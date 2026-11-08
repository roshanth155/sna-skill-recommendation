"""
Curated skill vocabulary, tokeniser and extractor for the co-occurrence network.

Why a curated list instead of raw phrases?
  The postings have no usable structured skill field (skills_desc is filled in ~1%
  of rows), so skills are read out of free-text descriptions. Free text contains
  unbounded "skill-ish" phrases, most of which are not skills. A fixed vocabulary
  gives a stable, auditable node set and avoids fuzzy matching, which would merge
  Java with JavaScript and C with C++ and silently corrupt the graph.

Why a token/ngram lookup instead of 240 regexes?
  One pass over the tokens per description is fast, and matching whole tokens means
  "sql" can never be found inside "mysql" and "java" never inside "javascript".
  Punctuation-bearing names (c++, c#, .net, node.js, ci/cd) are protected before
  tokenising so they survive as single tokens.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------- vocabulary
# canonical name -> (domain, [aliases])
# Domains are used later to ask "do communities align with skill domains?".
SKILLS: dict[str, tuple[str, list[str]]] = {
    # ---------------------------------------------------------- Programming
    "Python": ("Programming", ["python", "python3"]),
    "Java": ("Programming", ["java"]),
    "JavaScript": ("Programming", ["javascript", "js", "ecmascript"]),
    "TypeScript": ("Programming", ["typescript", "ts"]),
    "C++": ("Programming", ["c++", "cpp"]),
    "C#": ("Programming", ["c#", "csharp"]),
    "PHP": ("Programming", ["php"]),
    "Ruby": ("Programming", ["ruby", "ruby on rails", "rails"]),
    "Swift": ("Programming", ["swift"]),
    "Kotlin": ("Programming", ["kotlin"]),
    "Scala": ("Programming", ["scala"]),
    "Go": ("Programming", ["golang", "go programming"]),
    "Rust": ("Programming", ["rust"]),
    "R": ("Programming", ["r programming", "rstudio", "r language"]),
    "MATLAB": ("Programming", ["matlab"]),
    "Perl": ("Programming", ["perl"]),
    "Dart": ("Programming", ["dart"]),
    "Objective-C": ("Programming", ["objective-c", "objective c"]),
    "VBA": ("Programming", ["vba", "visual basic"]),
    "Shell Scripting": ("Programming", ["shell scripting", "shell script"]),
    # ----------------------------------------------------------------- Web
    "HTML": ("Web", ["html", "html5"]),
    "CSS": ("Web", ["css", "css3"]),
    "React": ("Web", ["react", "reactjs", "react js"]),
    "Angular": ("Web", ["angular", "angularjs"]),
    "Vue.js": ("Web", ["vuejs", "vue js", "vue"]),
    "Node.js": ("Web", ["nodejs", "node js"]),
    "Next.js": ("Web", ["nextjs", "next js"]),
    "jQuery": ("Web", ["jquery"]),
    "Bootstrap": ("Web", ["bootstrap"]),
    "Tailwind CSS": ("Web", ["tailwind", "tailwind css"]),
    "Redux": ("Web", ["redux"]),
    "GraphQL": ("Web", ["graphql"]),
    "REST API": ("Web", ["rest api", "restful", "restful api", "rest apis"]),
    "WordPress": ("Web", ["wordpress"]),
    "Shopify": ("Web", ["shopify"]),
    "Drupal": ("Web", ["drupal"]),
    "Web Accessibility": ("Web", ["wcag", "web accessibility", "accessibility standards"]),
    # -------------------------------------------------------------- Mobile
    "Android": ("Mobile", ["android"]),
    "iOS": ("Mobile", ["ios"]),
    "React Native": ("Mobile", ["react native"]),
    "Flutter": ("Mobile", ["flutter"]),
    "Xamarin": ("Mobile", ["xamarin"]),
    # ----------------------------------------------------------- Databases
    "SQL": ("Databases", ["sql", "structured query language"]),
    "MySQL": ("Databases", ["mysql"]),
    "PostgreSQL": ("Databases", ["postgresql", "postgres", "psql"]),
    "SQL Server": ("Databases", ["sql server", "ms sql", "mssql", "microsoft sql server", "t-sql", "tsql"]),
    "Oracle Database": ("Databases", ["oracle database", "oracle db", "plsql", "pl/sql"]),
    "MongoDB": ("Databases", ["mongodb", "mongo db"]),
    "Redis": ("Databases", ["redis"]),
    "Cassandra": ("Databases", ["cassandra"]),
    "Elasticsearch": ("Databases", ["elasticsearch", "elastic search"]),
    "DynamoDB": ("Databases", ["dynamodb"]),
    "SQLite": ("Databases", ["sqlite"]),
    "Snowflake": ("Databases", ["snowflake"]),
    "BigQuery": ("Databases", ["bigquery", "big query"]),
    "Redshift": ("Databases", ["redshift"]),
    "Hive": ("Databases", ["hive"]),
    "Neo4j": ("Databases", ["neo4j"]),
    "Firebase": ("Databases", ["firebase"]),
    # ------------------------------------------------------------ Data / ML
    "Machine Learning": ("Data/ML", ["machine learning", "ml models"]),
    "Deep Learning": ("Data/ML", ["deep learning", "neural networks"]),
    "NLP": ("Data/ML", ["natural language processing", "nlp"]),
    "Computer Vision": ("Data/ML", ["computer vision"]),
    "TensorFlow": ("Data/ML", ["tensorflow", "tensor flow"]),
    "PyTorch": ("Data/ML", ["pytorch", "torch"]),
    "Keras": ("Data/ML", ["keras"]),
    "scikit-learn": ("Data/ML", ["scikit-learn", "sklearn", "scikit learn"]),
    "XGBoost": ("Data/ML", ["xgboost"]),
    "pandas": ("Data/ML", ["pandas"]),
    "NumPy": ("Data/ML", ["numpy"]),
    "Apache Spark": ("Data/ML", ["apache spark", "spark", "pyspark"]),
    "Hadoop": ("Data/ML", ["hadoop", "apache hadoop"]),
    "Apache Kafka": ("Data/ML", ["kafka", "apache kafka"]),
    "Apache Airflow": ("Data/ML", ["airflow", "apache airflow"]),
    "dbt": ("Data/ML", ["dbt"]),
    "ETL": ("Data/ML", ["etl", "elt", "etl pipelines"]),
    "Data Warehousing": ("Data/ML", ["data warehousing", "data warehouse"]),
    "Data Modeling": ("Data/ML", ["data modeling", "data modelling"]),
    "Data Analysis": ("Data/ML", ["data analysis", "data analytics"]),
    "Data Visualization": ("Data/ML", ["data visualization", "data visualisation"]),
    "Statistics": ("Data/ML", ["statistics", "statistical analysis"]),
    "A/B Testing": ("Data/ML", ["a/b testing", "ab testing", "split testing"]),
    "Time Series Analysis": ("Data/ML", ["time series"]),
    "Data Quality": ("Data/ML", ["data quality", "data governance"]),
    "Big Data": ("Data/ML", ["big data"]),
    # ----------------------------------------------------------------- BI
    "Tableau": ("BI", ["tableau"]),
    "Power BI": ("BI", ["power bi", "powerbi"]),
    "Looker": ("BI", ["looker"]),
    "Google Analytics": ("BI", ["google analytics", "ga4"]),
    "SAS": ("BI", ["sas"]),
    "SPSS": ("BI", ["spss"]),
    "Microsoft Excel": ("BI", ["excel", "ms excel", "microsoft excel", "advanced excel"]),
    "Google Sheets": ("BI", ["google sheets"]),
    # ------------------------------------------------------- Cloud / DevOps
    "AWS": ("Cloud/DevOps", ["aws", "amazon web services"]),
    "Microsoft Azure": ("Cloud/DevOps", ["azure", "microsoft azure"]),
    "Google Cloud Platform": ("Cloud/DevOps", ["gcp", "google cloud", "google cloud platform"]),
    "Docker": ("Cloud/DevOps", ["docker"]),
    "Kubernetes": ("Cloud/DevOps", ["kubernetes", "k8s"]),
    "Terraform": ("Cloud/DevOps", ["terraform"]),
    "Ansible": ("Cloud/DevOps", ["ansible"]),
    "Jenkins": ("Cloud/DevOps", ["jenkins"]),
    "GitLab CI": ("Cloud/DevOps", ["gitlab ci", "gitlab"]),
    "GitHub Actions": ("Cloud/DevOps", ["github actions"]),
    "CI/CD": ("Cloud/DevOps", ["ci/cd", "cicd", "continuous integration", "continuous delivery"]),
    "Linux": ("Cloud/DevOps", ["linux"]),
    "Unix": ("Cloud/DevOps", ["unix"]),
    "Bash": ("Cloud/DevOps", ["bash"]),
    "PowerShell": ("Cloud/DevOps", ["powershell"]),
    "Nginx": ("Cloud/DevOps", ["nginx"]),
    "Grafana": ("Cloud/DevOps", ["grafana"]),
    "Prometheus": ("Cloud/DevOps", ["prometheus"]),
    "Splunk": ("Cloud/DevOps", ["splunk"]),
    "Git": ("Cloud/DevOps", ["git", "version control"]),
    "Infrastructure as Code": ("Cloud/DevOps", ["infrastructure as code", "iac"]),
    "Microservices": ("Cloud/DevOps", ["microservices"]),
    # ---------------------------------------------------------- IT / Support
    "Active Directory": ("IT/Support", ["active directory"]),
    "VMware": ("IT/Support", ["vmware"]),
    "Office 365": ("IT/Support", ["office 365", "microsoft 365", "o365"]),
    "SharePoint": ("IT/Support", ["sharepoint"]),
    "ServiceNow": ("IT/Support", ["servicenow"]),
    "Jira": ("IT/Support", ["jira"]),
    "Confluence": ("IT/Support", ["confluence"]),
    "ITIL": ("IT/Support", ["itil"]),
    "TCP/IP": ("IT/Support", ["tcp/ip", "tcp ip"]),
    "DNS": ("IT/Support", ["dns"]),
    "VPN": ("IT/Support", ["vpn"]),
    "Help Desk": ("IT/Support", ["help desk", "helpdesk", "technical support"]),
    "Hardware Troubleshooting": ("IT/Support", ["hardware troubleshooting", "troubleshooting hardware"]),
    # ------------------------------------------------------------ Security
    "Cybersecurity": ("Security", ["cybersecurity", "cyber security", "information security"]),
    "Penetration Testing": ("Security", ["penetration testing", "pen testing"]),
    "SIEM": ("Security", ["siem"]),
    "ISO 27001": ("Security", ["iso 27001"]),
    "Incident Response": ("Security", ["incident response"]),
    # --------------------------------------------------- Project / Business
    "Project Management": ("Project/Business", ["project management"]),
    "Agile": ("Project/Business", ["agile"]),
    "Scrum": ("Project/Business", ["scrum"]),
    "Kanban": ("Project/Business", ["kanban"]),
    "PMP": ("Project/Business", ["pmp", "project management professional"]),
    "Stakeholder Management": ("Project/Business", ["stakeholder management", "stakeholders"]),
    "Business Analysis": ("Project/Business", ["business analysis", "business analyst"]),
    "Requirements Gathering": ("Project/Business", ["requirements gathering", "requirements analysis"]),
    "Process Improvement": ("Project/Business", ["process improvement", "continuous improvement"]),
    "Six Sigma": ("Project/Business", ["six sigma"]),
    "SAP": ("Project/Business", ["sap"]),
    "ERP": ("Project/Business", ["erp"]),
    "CRM": ("Project/Business", ["crm"]),
    "Salesforce": ("Project/Business", ["salesforce"]),
    "HubSpot": ("Project/Business", ["hubspot"]),
    "Vendor Management": ("Project/Business", ["vendor management"]),
    "Contract Negotiation": ("Project/Business", ["contract negotiation"]),
    "Change Management": ("Project/Business", ["change management"]),
    # ------------------------------------------------------------- Finance
    "Financial Modeling": ("Finance", ["financial modeling", "financial modelling"]),
    "Forecasting": ("Finance", ["forecasting"]),
    "Budgeting": ("Finance", ["budgeting", "budget management"]),
    "Accounting": ("Finance", ["accounting"]),
    "Accounts Payable": ("Finance", ["accounts payable"]),
    "Accounts Receivable": ("Finance", ["accounts receivable"]),
    "Payroll": ("Finance", ["payroll"]),
    "GAAP": ("Finance", ["gaap"]),
    "IFRS": ("Finance", ["ifrs"]),
    "Auditing": ("Finance", ["auditing", "internal audit"]),
    "Tax Preparation": ("Finance", ["tax preparation", "tax returns"]),
    "Bookkeeping": ("Finance", ["bookkeeping"]),
    "Reconciliation": ("Finance", ["reconciliation", "reconciliations"]),
    "Variance Analysis": ("Finance", ["variance analysis"]),
    "Financial Reporting": ("Finance", ["financial reporting"]),
    "Risk Management": ("Finance", ["risk management"]),
    "Compliance": ("Finance", ["compliance"]),
    "Procurement": ("Finance", ["procurement"]),
    "Financial Analysis": ("Finance", ["financial analysis"]),
    # -------------------------------------------------------- Supply chain
    "Supply Chain": ("Supply chain", ["supply chain"]),
    "Inventory Management": ("Supply chain", ["inventory management", "inventory control"]),
    "Logistics": ("Supply chain", ["logistics"]),
    "Demand Planning": ("Supply chain", ["demand planning"]),
    "Quality Assurance": ("Supply chain", ["quality assurance", "qa"]),
    # ----------------------------------------------------- Sales / Marketing
    "Sales": ("Sales/Marketing", ["sales"]),
    "Business Development": ("Sales/Marketing", ["business development"]),
    "Customer Service": ("Sales/Marketing", ["customer service", "customer support"]),
    "Account Management": ("Sales/Marketing", ["account management"]),
    "Lead Generation": ("Sales/Marketing", ["lead generation", "prospecting"]),
    "Digital Marketing": ("Sales/Marketing", ["digital marketing"]),
    "SEO": ("Sales/Marketing", ["seo", "search engine optimization"]),
    "Email Marketing": ("Sales/Marketing", ["email marketing", "mailchimp"]),
    "Social Media Marketing": ("Sales/Marketing", ["social media marketing"]),
    "Content Marketing": ("Sales/Marketing", ["content marketing"]),
    "Copywriting": ("Sales/Marketing", ["copywriting"]),
    "Market Research": ("Sales/Marketing", ["market research"]),
    "Branding": ("Sales/Marketing", ["branding", "brand management"]),
    "E-commerce": ("Sales/Marketing", ["e-commerce", "ecommerce"]),
    "Cold Calling": ("Sales/Marketing", ["cold calling"]),
    # -------------------------------------------------------------- Design
    "Graphic Design": ("Design", ["graphic design"]),
    "Adobe Photoshop": ("Design", ["photoshop", "adobe photoshop"]),
    "Adobe Illustrator": ("Design", ["illustrator", "adobe illustrator"]),
    "Adobe InDesign": ("Design", ["indesign", "adobe indesign"]),
    "Figma": ("Design", ["figma"]),
    "UX Design": ("Design", ["ux design", "user experience"]),
    "UI Design": ("Design", ["ui design", "user interface design"]),
    "Wireframing": ("Design", ["wireframing", "wireframes"]),
    "Video Editing": ("Design", ["video editing"]),
    # ----------------------------------------------------------- HR / Admin
    "Recruiting": ("HR/Admin", ["recruiting", "recruitment"]),
    "Talent Acquisition": ("HR/Admin", ["talent acquisition"]),
    "Onboarding": ("HR/Admin", ["onboarding"]),
    "Employee Relations": ("HR/Admin", ["employee relations"]),
    "Performance Management": ("HR/Admin", ["performance management"]),
    "Training and Development": ("HR/Admin", ["training and development"]),
    "Data Entry": ("HR/Admin", ["data entry"]),
    "Office Administration": ("HR/Admin", ["office administration", "administrative support"]),
    "Event Planning": ("HR/Admin", ["event planning"]),
    "Scheduling": ("HR/Admin", ["scheduling"]),
    "Workforce Planning": ("HR/Admin", ["workforce planning"]),
    # --------------------------------------------------------- Healthcare
    "Patient Care": ("Healthcare", ["patient care"]),
    "Nursing": ("Healthcare", ["nursing", "registered nurse"]),
    "Electronic Health Records": ("Healthcare", ["electronic health records", "ehr", "emr"]),
    "HIPAA": ("Healthcare", ["hipaa"]),
    "Medical Coding": ("Healthcare", ["medical coding"]),
    "ICD-10": ("Healthcare", ["icd-10", "icd 10"]),
    "Phlebotomy": ("Healthcare", ["phlebotomy"]),
    "Clinical Research": ("Healthcare", ["clinical research", "clinical trials"]),
    # --------------------------------------------------------- Engineering
    "AutoCAD": ("Engineering", ["autocad"]),
    "SolidWorks": ("Engineering", ["solidworks"]),
    "PLC Programming": ("Engineering", ["plc programming", "programmable logic controller"]),
    "Lean Manufacturing": ("Engineering", ["lean manufacturing"]),
    "ISO 9001": ("Engineering", ["iso 9001"]),
    "Quality Control": ("Engineering", ["quality control", "qc"]),
    "GMP": ("Engineering", ["gmp", "good manufacturing practice"]),
    "OSHA": ("Engineering", ["osha"]),
    "Preventive Maintenance": ("Engineering", ["preventive maintenance"]),
    "HVAC": ("Engineering", ["hvac"]),
    "Welding": ("Engineering", ["welding"]),
    # ----------------------------------------------------------- Education
    "Curriculum Development": ("Education", ["curriculum development"]),
    "Lesson Planning": ("Education", ["lesson planning"]),
    "Classroom Management": ("Education", ["classroom management"]),
    # ---------------------------------------------------------- Soft skills
    "Communication": ("Soft skills", ["communication", "written communication", "verbal communication"]),
    "Leadership": ("Soft skills", ["leadership"]),
    "Teamwork": ("Soft skills", ["teamwork", "team player"]),
    "Problem Solving": ("Soft skills", ["problem solving"]),
    "Time Management": ("Soft skills", ["time management"]),
    "Critical Thinking": ("Soft skills", ["critical thinking"]),
    "Attention to Detail": ("Soft skills", ["attention to detail"]),
    "Organizational Skills": ("Soft skills", ["organizational skills", "organisation skills"]),
    "Presentation Skills": ("Soft skills", ["presentation skills", "public speaking"]),
    "Negotiation": ("Soft skills", ["negotiation", "negotiating"]),
    "Conflict Resolution": ("Soft skills", ["conflict resolution"]),
    "Mentoring": ("Soft skills", ["mentoring", "coaching"]),
    "Multitasking": ("Soft skills", ["multitasking", "multi-tasking"]),
}

# Skills so generic that a recommendation made only of them is useless.
# They stay in the graph (they are real demand signal) but are filtered out of the
# final Top-K recommendations unless paired with a specific skill.
GENERIC_SKILLS = {
    "Communication", "Leadership", "Teamwork", "Problem Solving", "Time Management",
    "Critical Thinking", "Attention to Detail", "Organizational Skills",
    "Presentation Skills", "Mentoring", "Multitasking", "Scheduling",
}

DOMAIN_OF = {name: domain for name, (domain, _) in SKILLS.items()}
DOMAINS = sorted({d for d, _ in SKILLS.values()})

# ------------------------------------------------------------- tokenisation
# Names whose punctuation would be destroyed by tokenisation. Replaced by a single
# alphanumeric token BEFORE stripping punctuation, on both the text and the aliases.
_PROTECTED = [
    ("asp.net", "aspnet"),
    (".net", "dotnet"),
    ("c++", "cplusplus"),
    ("c#", "csharp"),
    ("ci/cd", "cicd"),
    ("tcp/ip", "tcpip"),
    ("a/b", "ab"),
    ("pl/sql", "plsql"),
    ("node.js", "nodejs"),
    ("vue.js", "vuejs"),
    ("next.js", "nextjs"),
    ("objective-c", "objectivec"),
    ("scikit-learn", "scikitlearn"),
    ("power bi", "powerbi"),
    ("power-bi", "powerbi"),
]
_PROTECTED.sort(key=lambda p: -len(p[0]))

_NON_TOKEN = re.compile(r"[^a-z0-9+# ]+")


def tokenize(text: str) -> list[str]:
    """Lowercase, protect punctuation-bearing names, split into word tokens."""
    t = str(text).lower()
    for raw, token in _PROTECTED:
        if raw in t:
            t = t.replace(raw, token)
    return _NON_TOKEN.sub(" ", t).split()


# Canonical names that must NOT be used as match patterns on their own: they are
# ordinary English words / single letters and would match prose. Their real
# aliases below are the only way they are detected.
_NO_BARE_CANONICAL = {"Go", "R"}


def _build_lookup() -> tuple[dict[tuple[str, ...], str], int]:
    lookup: dict[tuple[str, ...], str] = {}
    for canonical, (_, aliases) in SKILLS.items():
        phrases = list(aliases) if canonical in _NO_BARE_CANONICAL else [canonical, *aliases]
        for phrase in phrases:
            key = tuple(tokenize(phrase))
            if not key:
                continue
            existing = lookup.get(key)
            if existing is not None and existing != canonical:
                raise ValueError(f"alias collision: {key} -> {existing} / {canonical}")
            lookup[key] = canonical
    longest = max(len(k) for k in lookup)
    return lookup, longest


LOOKUP, MAX_NGRAM = _build_lookup()

# "excel" is also a verb ("excel in a fast-paced environment"); these bigrams
# suppress that reading. Same idea for any other verb-like skill name.
NEGATIVE_BIGRAMS = {
    ("excel", "in"), ("excel", "at"), ("excel", "with"), ("excel", "as"),
    ("excel", "under"), ("excel", "during"), ("excel", "when"), ("excel", "by"),
}


def extract_skills(text: str) -> set[str]:
    """Return the canonical skills mentioned in one job description.

    Greedy longest-match over token n-grams: 'machine learning' is consumed as one
    match, so the span is not also counted as 'learning'. Each position advances,
    which keeps this linear in the number of tokens.
    """
    tokens = tokenize(text)
    n = len(tokens)
    found: set[str] = set()
    i = 0
    while i < n:
        matched = False
        for size in range(min(MAX_NGRAM, n - i), 0, -1):
            key = tuple(tokens[i:i + size])
            canonical = LOOKUP.get(key)
            if canonical is None:
                continue
            if size == 1 and i + 1 < n and key + (tokens[i + 1],) in NEGATIVE_BIGRAMS:
                continue
            found.add(canonical)
            i += size
            matched = True
            break
        if not matched:
            i += 1
    return found


if __name__ == "__main__":
    print(f"{len(SKILLS)} skills, {len(DOMAINS)} domains, "
          f"{len(LOOKUP)} alias phrases, max n-gram {MAX_NGRAM}")
    demo = (
        "We need a data analyst with strong SQL and Python, experience building "
        "Tableau dashboards, Excel modelling, A/B testing and stakeholder "
        "communication. C++ or C# is a plus; Kubernetes and CI/CD preferred. "
        "You excel in a fast-paced environment and know Node.js and React."
    )
    print(sorted(extract_skills(demo)))
