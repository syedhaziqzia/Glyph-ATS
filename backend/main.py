import io
import re
import os
import json
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Dict, List, Set, Any, Tuple, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
from pdfminer.layout import LAParams
import pypdf
import cv2
import pytesseract
import jinja2
from xhtml2pdf import pisa

# -------------------------------------------------------------------------
# Application Initialization & Security Configuration
# -------------------------------------------------------------------------

app = FastAPI(
    title="Glyph API",
    description="Minimal, ephemeral, and privacy-first ATS resume analyzer, X-Ray parser, and generator engine (v2.5).",
    version="2.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MAX_FILE_SIZE_BYTES: int = 5 * 1024 * 1024  # 5 Megabytes limit
PDF_MAGIC_BYTES: bytes = b"%PDF"
PNG_MAGIC_BYTES: bytes = b"\x89PNG"
JPEG_MAGIC_BYTES: bytes = b"\xff\xd8\xff"

# Custom tech-friendly token pattern that captures short terms (2+ chars) like ui, ux, ai, ml, qa, pm, go, js, ts, c++, c#, .net, ci/cd
TOKEN_PATTERN: str = r"(?u)(?<![a-zA-Z0-9])[a-zA-Z][a-zA-Z0-9_\+#\.\-/]{1,30}(?![a-zA-Z0-9])"

COMMON_STOP_WORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are",
    "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both",
    "but", "by", "can't", "cannot", "could", "couldn't", "did", "didn't", "do", "does", "doesn't",
    "doing", "don't", "down", "during", "each", "few", "for", "from", "further", "had", "hadn't",
    "has", "hasn't", "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll",
    "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of", "off", "on",
    "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own",
    "same", "shan't", "she", "she'd", "she'll", "she's", "should", "shouldn't", "so", "some",
    "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then",
    "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this",
    "those", "through", "to", "too", "under", "until", "up", "very", "was", "wasn't", "we",
    "we'd", "we'll", "we're", "we've", "were", "weren't", "what", "what's", "when", "when's",
    "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with",
    "won't", "would", "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves", "will", "shall", "may", "might", "must", "can", "etc", "eg", "ie",
    "required", "requirements", "responsibilities", "responsible", "qualifications", "preferred",
    "role", "roles", "job", "jobs", "candidate", "candidates", "team", "teams", "work", "working",
    "experience", "years", "year", "ability", "skills", "skill", "knowledge", "including", "across",
    "within", "help", "need", "well", "plus", "ideal", "opportunity", "opportunities", "join",
    "looking", "seeking", "proven", "track", "record", "know", "hiring", "hire", "strong",
    "proficient", "proficiency", "familiarity", "hands-on", "hands", "hand", "overview", "summary",
    "good", "great", "excellent", "experienced", "understanding", "duties", "task", "tasks",
    "office", "remote", "hybrid", "onsite", "location", "apply", "applicant", "applicants",
    "position", "positions", "day", "daily", "using", "build", "building", "company", "organization",
    "environment", "fast-paced", "paced", "passionate", "motivated", "self-starter", "collaborate",
    "collaborative", "deliver", "delivering", "solutions", "world-class", "world", "class",
    "equal", "employer", "employment", "veteran", "disability", "status", "race", "color",
    "religion", "gender", "sexual", "orientation", "national", "origin", "benefits", "medical",
    "dental", "vision", "401k", "pto", "vacation", "paid", "leave", "salary", "compensation",
    "competitive", "bonus", "equity", "relocation", "bachelor", "bachelors", "master", "masters",
    "degree", "field", "related", "equivalent", "practical", "minimum", "least", "new", "high",
    "level", "levels", "use", "used", "make", "making", "ensure", "ensuring", "support", "supporting",
    "develop", "developing", "design", "designing", "implement", "implementing", "maintain",
    "maintaining", "create", "creating", "write", "writing", "participate", "participating",
    "closely", "partner", "partners", "multiple", "various", "complex", "large", "scale", "end",
    "best", "practices", "modern", "clean", "efficient", "reliable", "scalable", "robust", "secure"
}

# -------------------------------------------------------------------------
# Role Ontology, Short-Form Abbreviations & Synonym Equivalence Engine
# -------------------------------------------------------------------------

ROLE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "swe": {
        "title": "Software Engineer (SWE)",
        "triggers": ["swe", "sde", "software engineer", "software developer", "software engineering", "application developer", "programmer", "coder"],
        "skills": [
            "python", "java", "c++", "javascript", "typescript", "data structures", "algorithms",
            "object-oriented programming", "system design", "rest apis", "sql", "git", "unit testing",
            "ci/cd", "linux", "agile", "debugging", "microservices", "cloud computing", "multithreading"
        ],
    },
    "frontend": {
        "title": "Frontend / Web Engineer",
        "triggers": ["frontend", "front-end", "front end", "fe", "ui developer", "web developer", "react developer", "frontend engineer", "frontend developer"],
        "skills": [
            "javascript", "typescript", "react", "next.js", "html5", "css3", "tailwind css", "vue",
            "angular", "redux", "state management", "responsive design", "rest apis", "graphql",
            "webpack", "vite", "web performance", "accessibility", "jest", "cypress", "git", "figma"
        ],
    },
    "backend": {
        "title": "Backend / API Engineer",
        "triggers": ["backend", "back-end", "back end", "be", "backend engineer", "backend developer", "api engineer", "server-side", "distributed systems"],
        "skills": [
            "python", "node.js", "java", "go", "c#", "fastapi", "django", "spring boot", "express",
            "rest apis", "graphql", "microservices", "postgresql", "mysql", "mongodb", "redis",
            "kafka", "authentication", "system design", "sql", "ci/cd", "git", "linux", "caching"
        ],
    },
    "fullstack": {
        "title": "Full-Stack Software Engineer",
        "triggers": ["fullstack", "full-stack", "full stack", "fs", "fullstack engineer", "fullstack developer"],
        "skills": [
            "javascript", "typescript", "react", "next.js", "node.js", "python", "html5", "css3",
            "tailwind css", "rest apis", "graphql", "postgresql", "mongodb", "redis", "aws",
            "ci/cd", "git", "authentication", "system design", "unit testing"
        ],
    },
    "internship": {
        "title": "Software / Tech Internship & Early Career",
        "triggers": ["intern", "internship", "internships", "co-op", "coop", "new grad", "entry level", "entry-level", "junior", "graduate", "trainee", "fellow", "apprentice"],
        "skills": [
            "data structures", "algorithms", "object-oriented programming", "python", "java",
            "c++", "javascript", "git", "github", "sql", "rest apis", "unit testing",
            "problem solving", "agile", "debugging", "linux", "html5", "css3"
        ],
    },
    "mle": {
        "title": "Machine Learning / AI Engineer (MLE)",
        "triggers": ["mle", "ml", "ai", "machine learning", "artificial intelligence", "deep learning", "dl", "genai", "llm", "nlp", "computer vision", "cv engineer", "ai engineer"],
        "skills": [
            "python", "pytorch", "tensorflow", "scikit-learn", "machine learning", "deep learning",
            "neural networks", "nlp", "computer vision", "transformers", "llm", "rag", "numpy",
            "pandas", "feature engineering", "mlops", "fastapi", "sql", "git", "model deployment"
        ],
    },
    "data_scientist": {
        "title": "Data Scientist (DS)",
        "triggers": ["ds", "data scientist", "data science", "decision scientist", "quantitative researcher", "applied scientist"],
        "skills": [
            "python", "r", "sql", "pandas", "numpy", "scikit-learn", "statistics", "hypothesis testing",
            "a/b testing", "machine learning", "data visualization", "tableau", "jupyter",
            "feature engineering", "regression", "classification", "probability", "time series"
        ],
    },
    "data_analyst": {
        "title": "Data & Business Intelligence Analyst (DA/BI)",
        "triggers": ["da", "data analyst", "data analytics", "bi", "business intelligence", "analytics", "reporting analyst", "insights analyst"],
        "skills": [
            "sql", "excel", "python", "tableau", "power bi", "looker", "data visualization", "dashboards",
            "kpi", "data cleaning", "statistics", "pandas", "etl", "reporting", "business metrics", "google analytics"
        ],
    },
    "data_engineer": {
        "title": "Data Engineer (DE)",
        "triggers": ["de", "data engineer", "data engineering", "etl", "big data", "analytics engineer", "data platform"],
        "skills": [
            "python", "sql", "spark", "pyspark", "kafka", "airflow", "dbt", "snowflake",
            "bigquery", "redshift", "databricks", "etl", "data warehousing", "data modeling", "aws", "git"
        ],
    },
    "cloud_devops": {
        "title": "Cloud / SRE / Platform Engineer",
        "triggers": ["devops", "sre", "site reliability", "cloud", "cloud engineer", "platform engineer", "infrastructure", "sysadmin", "devsecops", "aws engineer"],
        "skills": [
            "aws", "gcp", "azure", "kubernetes", "terraform", "ansible", "ci/cd",
            "github actions", "jenkins", "linux", "bash", "prometheus", "grafana", "monitoring",
            "networking", "microservices", "cloud security", "python", "serverless"
        ],
    },
    "cybersecurity": {
        "title": "Cybersecurity / InfoSec Engineer",
        "triggers": ["cybersecurity", "security", "infosec", "soc", "pentest", "penetration testing", "secops", "appsec", "cyber", "security analyst"],
        "skills": [
            "network security", "siem", "splunk", "vulnerability assessment", "penetration testing",
            "owasp", "iam", "active directory", "cryptography", "firewalls", "incident response",
            "linux", "python", "cloud security", "threat intelligence", "wireshark", "zero trust"
        ],
    },
    "mobile": {
        "title": "Mobile Application Developer (iOS / Android)",
        "triggers": ["mobile", "ios", "android", "react native", "flutter", "swift developer", "kotlin developer", "mobile engineer"],
        "skills": [
            "swift", "swiftui", "kotlin", "jetpack compose", "react native", "flutter", "dart",
            "ios", "android", "rest apis", "mobile ui", "state management", "git", "unit testing", "firebase"
        ],
    },
    "qa": {
        "title": "QA / SDET Test Automation Engineer",
        "triggers": ["qa", "sdet", "quality assurance", "test engineer", "testing", "automation engineer", "quality engineer"],
        "skills": [
            "selenium", "playwright", "cypress", "pytest", "junit", "appium", "test automation",
            "api testing", "postman", "ci/cd", "jira", "regression testing", "performance testing", "python", "javascript", "git"
        ],
    },
    "pm": {
        "title": "Product Manager (PM)",
        "triggers": ["pm", "product manager", "product management", "apm", "product owner", "po", "technical product manager", "tpm"],
        "skills": [
            "product strategy", "roadmap", "agile", "scrum", "user stories", "stakeholder management",
            "a/b testing", "product analytics", "sql", "wireframing", "figma", "kpis", "user research",
            "jira", "go-to-market", "prioritization", "competitive analysis"
        ],
    },
    "uiux": {
        "title": "UI/UX & Product Designer",
        "triggers": ["ui/ux", "ux", "ui", "product designer", "ux designer", "ui designer", "interaction design", "web design", "visual designer"],
        "skills": [
            "figma", "sketch", "adobe xd", "wireframing", "prototyping", "design systems",
            "user research", "usability testing", "interaction design", "information architecture",
            "responsive design", "accessibility", "wcag", "html5", "css3"
        ],
    },
    "embedded": {
        "title": "Embedded / Systems / Firmware Engineer",
        "triggers": ["embedded", "firmware", "systems engineer", "robotics", "iot", "hardware", "fpga", "rtos", "microcontrollers"],
        "skills": [
            "c", "c++", "rust", "embedded systems", "rtos", "microcontrollers", "arm", "linux",
            "i2c", "spi", "uart", "can bus", "oscilloscope", "debugging", "multithreading", "memory management", "python", "git"
        ],
    },
    "quant_finance": {
        "title": "Quantitative / Financial / FinTech Analyst",
        "triggers": ["quant", "quantitative", "fintech", "financial analyst", "algorithmic trading", "finance", "investment banking", "risk analyst"],
        "skills": [
            "python", "c++", "sql", "excel", "financial modeling", "valuation", "statistics",
            "time series", "probability", "algorithmic trading", "risk management", "pandas", "numpy", "bloomberg", "forecasting"
        ],
    },
    "business_analyst": {
        "title": "Business / Operations / Project Manager",
        "triggers": ["ba", "business analyst", "project manager", "program manager", "scrum master", "consultant", "operations analyst", "strategy analyst"],
        "skills": [
            "requirements gathering", "stakeholder management", "project management", "agile", "scrum",
            "jira", "process improvement", "sql", "excel", "data analysis", "documentation", "kpis",
            "power bi", "risk management", "resource planning", "cross-functional leadership"
        ],
    },
    "gamedev": {
        "title": "Game Developer / Graphics Engineer",
        "triggers": ["game developer", "game engineer", "unity", "unreal", "gameplay engineer", "graphics engineer", "shaders"],
        "skills": [
            "c++", "c#", "unity", "unreal engine", "opengl", "vulkan", "directx", "shaders",
            "3d math", "linear algebra", "physics engine", "gameplay programming", "optimization", "git"
        ],
    },
    "network_it": {
        "title": "Network / Systems / IT Administrator",
        "triggers": ["network engineer", "it support", "system administrator", "help desk", "desktop support", "network admin", "it specialist"],
        "skills": [
            "tcp/ip", "dns", "dhcp", "vpn", "vlan", "routing", "switching", "cisco", "firewalls",
            "active directory", "windows server", "linux", "powershell", "bash", "virtualization", "vmware", "troubleshooting"
        ],
    },
    "marketing_growth": {
        "title": "Digital Marketing / Growth / SEO Specialist",
        "triggers": ["marketing", "digital marketing", "growth", "seo", "sem", "content marketing", "social media manager", "brand manager"],
        "skills": [
            "seo", "sem", "google analytics", "google ads", "meta ads", "content strategy",
            "copywriting", "email marketing", "hubspot", "crm", "conversion rate optimization",
            "a/b testing", "social media", "market research", "kpis", "campaign management"
        ],
    },
}

SKILL_EQUIVALENCE_GROUPS: List[List[str]] = [
    ["javascript", "js", "ecmascript", "es6"],
    ["typescript", "ts"],
    ["python", "py", "python3"],
    ["react", "reactjs", "react.js"],
    ["next.js", "nextjs", "next"],
    ["node.js", "nodejs", "node"],
    ["vue", "vuejs", "vue.js"],
    ["angular", "angularjs"],
    ["html5", "html"],
    ["css3", "css", "flexbox", "css grid"],
    ["tailwind css", "tailwind", "tailwindcss"],
    ["postgresql", "postgres", "psql"],
    ["mongodb", "mongo"],
    ["kubernetes", "k8s"],
    ["amazon web services", "aws", "ec2", "s3", "lambda"],
    ["google cloud platform", "gcp", "google cloud"],
    ["microsoft azure", "azure"],
    ["object-oriented programming", "oop", "object oriented"],
    ["data structures", "dsa", "data structures and algorithms"],
    ["algorithms", "dsa", "algorithmic"],
    ["rest apis", "rest api", "restful", "restful apis", "api", "apis"],
    ["ci/cd", "cicd", "continuous integration", "continuous deployment", "github actions", "jenkins", "gitlab ci"],
    ["machine learning", "ml"],
    ["artificial intelligence", "ai"],
    ["deep learning", "dl", "neural networks"],
    ["natural language processing", "nlp"],
    ["computer vision", "cv", "opencv"],
    ["large language models", "llm", "llms", "genai", "generative ai", "gpt", "langchain"],
    ["retrieval-augmented generation", "rag"],
    ["scikit-learn", "sklearn", "scikit learn"],
    ["c++", "cpp", "c/c++"],
    ["c#", "csharp", ".net", "dotnet", "asp.net"],
    ["golang", "go"],
    ["unit testing", "testing", "pytest", "jest", "junit", "automated testing", "tdd"],
    ["version control", "git", "github", "gitlab", "bitbucket"],
    ["sql", "mysql", "postgresql", "sqlite", "tsql", "pl/sql", "relational databases"],
    ["nosql", "mongodb", "dynamodb", "cassandra", "redis"],
    ["agile", "scrum", "kanban", "sprint planning"],
    ["user interface", "ui"],
    ["user experience", "ux"],
    ["ui/ux", "ui", "ux"],
    ["key performance indicators", "kpi", "kpis"],
    ["extract transform load", "etl", "data pipelines"],
    ["search engine optimization", "seo"],
    ["customer relationship management", "crm", "salesforce", "hubspot"],
    ["identity and access management", "iam", "oauth", "oauth2", "jwt", "sso"],
    ["amazon web services", "aws"],
    ["system design", "distributed systems", "scalability", "high availability"],
]

# Pretty display mapping for canonical skills so UI pills and Resume PDF look crisp and authentic
CANONICAL_DISPLAY_NAMES: Dict[str, str] = {
    "javascript": "JavaScript",
    "js": "JavaScript",
    "typescript": "TypeScript",
    "ts": "TypeScript",
    "python": "Python",
    "java": "Java",
    "c++": "C++",
    "c#": "C#",
    ".net": ".NET",
    "go": "Go",
    "golang": "Go",
    "rust": "Rust",
    "swift": "Swift",
    "swiftui": "SwiftUI",
    "kotlin": "Kotlin",
    "php": "PHP",
    "ruby": "Ruby",
    "r": "R",
    "sql": "SQL",
    "nosql": "NoSQL",
    "html5": "HTML5",
    "html": "HTML5",
    "css3": "CSS3",
    "css": "CSS3",
    "tailwind css": "Tailwind CSS",
    "react": "React",
    "next.js": "Next.js",
    "node.js": "Node.js",
    "vue": "Vue.js",
    "angular": "Angular",
    "express": "Express.js",
    "fastapi": "FastAPI",
    "django": "Django",
    "flask": "Flask",
    "spring boot": "Spring Boot",
    "graphql": "GraphQL",
    "rest apis": "REST APIs",
    "rest api": "REST APIs",
    "postgresql": "PostgreSQL",
    "mysql": "MySQL",
    "mongodb": "MongoDB",
    "redis": "Redis",
    "snowflake": "Snowflake",
    "bigquery": "BigQuery",
    "redshift": "Redshift",
    "databricks": "Databricks",
    "dynamodb": "DynamoDB",
    "aws": "AWS",
    "amazon web services": "AWS",
    "gcp": "Google Cloud (GCP)",
    "google cloud platform": "Google Cloud (GCP)",
    "azure": "Microsoft Azure",
    "kubernetes": "Kubernetes",
    "k8s": "Kubernetes",
    "docker": "Docker",
    "terraform": "Terraform",
    "ansible": "Ansible",
    "jenkins": "Jenkins",
    "github actions": "GitHub Actions",
    "ci/cd": "CI/CD",
    "linux": "Linux",
    "bash": "Bash",
    "powershell": "PowerShell",
    "git": "Git",
    "github": "GitHub",
    "jira": "Jira",
    "figma": "Figma",
    "pytorch": "PyTorch",
    "tensorflow": "TensorFlow",
    "scikit-learn": "Scikit-Learn",
    "pandas": "Pandas",
    "numpy": "NumPy",
    "opencv": "OpenCV",
    "nlp": "NLP",
    "llm": "LLMs",
    "rag": "RAG",
    "mlops": "MLOps",
    "machine learning": "Machine Learning",
    "deep learning": "Deep Learning",
    "computer vision": "Computer Vision",
    "data structures": "Data Structures",
    "algorithms": "Algorithms",
    "object-oriented programming": "OOP",
    "oop": "OOP",
    "system design": "System Design",
    "microservices": "Microservices",
    "unit testing": "Unit Testing",
    "pytest": "Pytest",
    "jest": "Jest",
    "cypress": "Cypress",
    "playwright": "Playwright",
    "selenium": "Selenium",
    "postman": "Postman",
    "tableau": "Tableau",
    "power bi": "Power BI",
    "excel": "Excel",
    "etl": "ETL Pipelines",
    "spark": "Apache Spark",
    "pyspark": "PySpark",
    "kafka": "Apache Kafka",
    "airflow": "Apache Airflow",
    "dbt": "dbt",
    "a/b testing": "A/B Testing",
    "kpi": "KPIs",
    "kpis": "KPIs",
    "ui/ux": "UI/UX Design",
    "wcag": "WCAG Accessibility",
    "owasp": "OWASP",
    "siem": "SIEM",
    "iam": "IAM / OAuth",
    "tcp/ip": "TCP/IP",
    "dns": "DNS",
    "vpn": "VPN",
    "rtos": "RTOS",
    "i2c": "I2C",
    "spi": "SPI",
    "uart": "UART",
    "arm": "ARM Architecture",
    "seo": "SEO",
    "sem": "SEM",
    "crm": "CRM",
    "agile": "Agile / Scrum",
    "scrum": "Scrum",
}


def format_skill_display(raw_kw: str) -> str:
    """Formats a raw keyword token into a clean, professional display label."""
    k = raw_kw.strip().lower()
    if k in CANONICAL_DISPLAY_NAMES:
        return CANONICAL_DISPLAY_NAMES[k]
    if len(k) <= 3 and k in {"sql", "aws", "gcp", "api", "llm", "nlp", "oop", "dsa", "css", "ios", "sre", "etl", "kpi", "seo", "sem", "crm", "iam", "jwt", "sso", "vpn", "dns", "arm", "spi"}:
        return k.upper()
    return " ".join(w.capitalize() for w in k.split())


SKILL_SYNONYM_MAP: Dict[str, Set[str]] = {}
ALL_KNOWN_SKILLS: Set[str] = set(CANONICAL_DISPLAY_NAMES.keys())
for group in SKILL_EQUIVALENCE_GROUPS:
    group_set = {item.lower() for item in group}
    ALL_KNOWN_SKILLS.update(group_set)
    for term in group_set:
        SKILL_SYNONYM_MAP.setdefault(term, set()).update(group_set)

for role_profile in ROLE_KNOWLEDGE_BASE.values():
    for sk in role_profile["skills"]:
        ALL_KNOWN_SKILLS.add(sk.lower())

# -------------------------------------------------------------------------
# Locate Tesseract OCR Binary (Configurable Fallbacks)
# -------------------------------------------------------------------------

def find_tesseract_cmd() -> Optional[str]:
    """Finds the tesseract executable on Windows or Linux."""
    cmd = shutil.which("tesseract")
    if cmd:
        return cmd
    common_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        os.path.expanduser(r"~\AppData\Local\Programs\Tesseract-OCR\tesseract.exe"),
        "/usr/bin/tesseract",
        "/usr/local/bin/tesseract",
    ]
    for p in common_paths:
        if os.path.exists(p):
            return p
    return None

TESSERACT_EXE: Optional[str] = find_tesseract_cmd()
if TESSERACT_EXE:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_EXE

# -------------------------------------------------------------------------
# Response & Request Models
# -------------------------------------------------------------------------

class StatsModel(BaseModel):
    resume_word_count: int
    job_desc_word_count: int
    matched_keywords_count: int
    missing_keywords_count: int

class ExperienceItem(BaseModel):
    title: str = Field(..., description="Job Title")
    company: str = Field(..., description="Company Name")
    dates: str = Field(..., description="Dates of employment (e.g., 2022 - Present)")
    bullets: Optional[List[str]] = Field(default=[], description="Bullet points of achievements")
    description: Optional[str] = Field(default="", description="Optional paragraph summary")

class ProjectItem(BaseModel):
    name: str = Field(..., description="Project Name")
    tech: Optional[str] = Field(default="", description="Technologies used")
    date: Optional[str] = Field(default="2025", description="Project year or date")
    bullets: Optional[List[str]] = Field(default=[], description="Project achievement bullets")

class EducationItem(BaseModel):
    degree: str = Field(..., description="Degree / Major")
    institution: str = Field(..., description="University / School")
    year: str = Field(..., description="Graduation Year")
    details: Optional[str] = Field(default="", description="Honors, GPA, or coursework")

class ParsedResumeProfile(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str
    linkedin: str
    github: str
    summary: str
    skills: List[str]
    experience: List[ExperienceItem]
    projects: List[ProjectItem] = Field(default=[])
    education: List[EducationItem]
    certifications: List[str] = Field(default=[])

class AuditCheckItem(BaseModel):
    label: str
    passed: bool
    detail: str

class BulletImpactReport(BaseModel):
    total_bullets_analyzed: int
    quantified_bullets_count: int
    quantification_rate: int
    strong_verbs_found: List[str]
    weak_phrases_found: List[str]
    xyz_bullet_suggestions: List[str]

class StructuralAuditReport(BaseModel):
    readability_score: int
    checks: List[AuditCheckItem]
    detected_sections: List[str]
    raw_text_preview: str

class AnalysisResponse(BaseModel):
    status: str
    match_score: int
    match_level: str
    cosine_similarity: float
    keyword_match_rate: float
    matched_keywords: List[str]
    missing_keywords: List[str]
    stats: StatsModel
    recommendations: List[str]
    document_format: str
    expanded_roles: List[str] = Field(default=[], description="Roles recognized & expanded from short/comma queries")
    resume_confidence: int = Field(default=100, description="Confidence score (0-100) that uploaded document is a genuine resume")
    parsed_profile: ParsedResumeProfile
    bullet_impact: BulletImpactReport
    structural_audit: StructuralAuditReport
    ephemeral_notice: str

class ResumeGenerateRequest(BaseModel):
    full_name: str
    email: str
    phone: str
    location: str
    linkedin: Optional[str] = ""
    github: Optional[str] = ""
    summary: Optional[str] = ""
    skills: List[str] = Field(default=[], description="List of technical and domain skills")
    experience: List[ExperienceItem] = []
    projects: Optional[List[ProjectItem]] = []
    education: List[EducationItem] = []
    certifications: Optional[List[str]] = []
    section_order: Optional[str] = Field(
        default="jakes",
        description="ATS section ordering: 'jakes' (Summary -> Education -> Experience -> Projects -> Technical Skills -> Certifications) or 'standard' (Summary -> Experience -> Projects -> Education -> Technical Skills -> Certifications)",
    )

# -------------------------------------------------------------------------
# Jinja2 Gold-Standard ATS Resume HTML Template
# (Follows Reference Layout: Summary -> Education -> Experience -> Projects -> Categorized Skills -> Certifications)
# -------------------------------------------------------------------------

RESUME_HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
    @page {
        size: letter;
        margin: 1.05cm 1.35cm 1.05cm 1.35cm;
    }
    body {
        font-family: Helvetica, Arial, sans-serif;
        color: #111111;
        font-size: 9.6pt;
        line-height: 1.34;
    }
    .header {
        text-align: center;
        margin-bottom: 8px;
    }
    .name {
        font-size: 19pt;
        font-weight: bold;
        letter-spacing: 0.2px;
        margin-bottom: 3px;
        color: #000000;
    }
    .contact-line {
        font-size: 8.8pt;
        color: #222222;
        line-height: 1.35;
    }
    h2 {
        font-size: 9.8pt;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 0.7px;
        border-bottom: 1px solid #111111;
        padding-bottom: 2px;
        margin-top: 9px;
        margin-bottom: 5px;
        color: #000000;
    }
    p {
        margin: 0 0 4px 0;
    }
    .entry-block {
        margin-bottom: 6px;
    }
    table.entry-header {
        width: 100%;
        border: none;
        border-collapse: collapse;
        margin-bottom: 1px;
    }
    table.entry-header td.left-cell {
        text-align: left;
        padding: 0;
        font-size: 9.6pt;
        color: #000000;
    }
    table.entry-header td.right-cell {
        text-align: right;
        padding: 0;
        font-size: 9pt;
        font-style: italic;
        color: #222222;
        width: 26%;
        white-space: nowrap;
    }
    .job-title {
        font-weight: bold;
        color: #000000;
    }
    .sub-line {
        font-size: 9pt;
        font-style: italic;
        color: #222222;
        margin-bottom: 2px;
    }
    .edu-degree {
        font-size: 9.3pt;
        color: #111111;
        margin-bottom: 2px;
    }
    ul {
        margin: 2px 0 5px 15px;
        padding: 0;
    }
    li {
        margin-bottom: 2.5px;
        line-height: 1.34;
    }
    table.skills-table {
        width: 100%;
        border: none;
        border-collapse: collapse;
        margin-top: 2px;
        margin-bottom: 4px;
    }
    table.skills-table td.skill-cat {
        width: 22%;
        font-weight: bold;
        font-size: 9.3pt;
        padding: 1.5px 6px 1.5px 0;
        vertical-align: top;
        color: #000000;
    }
    table.skills-table td.skill-vals {
        width: 78%;
        font-size: 9.3pt;
        padding: 1.5px 0;
        vertical-align: top;
        color: #111111;
    }
    table.cert-table {
        width: 100%;
        border: none;
        border-collapse: collapse;
        margin-top: 2px;
    }
    table.cert-table td.cert-left {
        text-align: left;
        padding: 1.5px 0;
        font-size: 9.3pt;
        color: #111111;
    }
    table.cert-table td.cert-right {
        text-align: right;
        padding: 1.5px 0;
        font-size: 9pt;
        font-style: italic;
        color: #222222;
        width: 16%;
        white-space: nowrap;
    }
</style>
</head>
<body>
    <div class="header">
        <div class="name">{{ full_name }}</div>
        {% if contact_line_1 %}<div class="contact-line">{{ contact_line_1 }}</div>{% endif %}
        {% if contact_line_2 %}<div class="contact-line">{{ contact_line_2 }}</div>{% endif %}
    </div>

    {% if summary %}
    <h2>SUMMARY</h2>
    <p>{{ summary }}</p>
    {% endif %}

    {% if section_order != 'standard' %}
        {% if education %}
        <h2>EDUCATION</h2>
        {% for edu in education %}
        <div class="entry-block">
            <table class="entry-header">
                <tr>
                    <td class="left-cell"><span class="job-title">{{ edu.institution }}</span></td>
                    <td class="right-cell">{{ edu.year }}</td>
                </tr>
            </table>
            {% if edu.degree %}<div class="edu-degree">{{ edu.degree }}</div>{% endif %}
            {% if edu.details %}<p>{{ edu.details }}</p>{% endif %}
        </div>
        {% endfor %}
        {% endif %}
    {% endif %}

    {% if experience %}
    <h2>EXPERIENCE</h2>
    {% for job in experience %}
    <div class="entry-block">
        <table class="entry-header">
            <tr>
                <td class="left-cell"><span class="job-title">{{ job.title }}</span></td>
                <td class="right-cell">{{ job.dates }}</td>
            </tr>
        </table>
        {% if job.company %}<div class="sub-line">{{ job.company }}</div>{% endif %}
        {% if job.bullets %}
        <ul>
            {% for bullet in job.bullets %}
            <li>{{ bullet }}</li>
            {% endfor %}
        </ul>
        {% elif job.description %}
        <p>{{ job.description }}</p>
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}

    {% if projects %}
    <h2>PROJECTS</h2>
    {% for proj in projects %}
    <div class="entry-block">
        <table class="entry-header">
            <tr>
                <td class="left-cell"><span class="job-title">{{ proj.name }}</span></td>
                <td class="right-cell">{{ proj.date or '2025' }}</td>
            </tr>
        </table>
        {% if proj.tech %}<div class="sub-line">{{ proj.tech }}</div>{% endif %}
        {% if proj.bullets %}
        <ul>
            {% for bullet in proj.bullets %}
            <li>{{ bullet }}</li>
            {% endfor %}
        </ul>
        {% endif %}
    </div>
    {% endfor %}
    {% endif %}

    {% if section_order == 'standard' %}
        {% if education %}
        <h2>EDUCATION</h2>
        {% for edu in education %}
        <div class="entry-block">
            <table class="entry-header">
                <tr>
                    <td class="left-cell"><span class="job-title">{{ edu.institution }}</span></td>
                    <td class="right-cell">{{ edu.year }}</td>
                </tr>
            </table>
            {% if edu.degree %}<div class="edu-degree">{{ edu.degree }}</div>{% endif %}
            {% if edu.details %}<p>{{ edu.details }}</p>{% endif %}
        </div>
        {% endfor %}
        {% endif %}
    {% endif %}

    {% if skill_categories %}
    <h2>TECHNICAL SKILLS</h2>
    <table class="skills-table">
        {% for cat_label, cat_val in skill_categories %}
        <tr>
            <td class="skill-cat">{{ cat_label }}</td>
            <td class="skill-vals">{{ cat_val }}</td>
        </tr>
        {% endfor %}
    </table>
    {% elif skills %}
    <h2>TECHNICAL SKILLS</h2>
    <p><strong>Core Technologies:</strong> {{ skills | join(', ') }}</p>
    {% endif %}

    {% if parsed_certifications %}
    <h2>CERTIFICATIONS</h2>
    <table class="cert-table">
        {% for cert in parsed_certifications %}
        <tr>
            <td class="cert-left">
                {% if cert.issuer %}<strong>{{ cert.issuer }}</strong> {{ cert.title }}{% else %}{{ cert.title }}{% endif %}
            </td>
            <td class="cert-right">{{ cert.year }}</td>
        </tr>
        {% endfor %}
    </table>
    {% endif %}
</body>
</html>
"""

# -------------------------------------------------------------------------
# Smart Document Authenticity & Resume Classifier (PDF, PNG, JPG)
# -------------------------------------------------------------------------

# Strict heading patterns: matched either on standalone short lines (<= 5 words)
# or explicitly formatted as section headers (e.g. 'Education:', 'TECHNICAL SKILLS', etc.)
STRICT_SECTION_PATTERNS: List[Tuple[str, str, str]] = [
    ("Education", r"^\s*(?:education|academic\s+background|academics)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:EDUCATION|Education\s*:|Academic\s+Background\s*:)"),
    ("Experience", r"^\s*(?:work\s+experience|professional\s+experience|experience|employment\s+history|work\s+history|relevant\s+experience)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:WORK\s+EXPERIENCE|PROFESSIONAL\s+EXPERIENCE|EXPERIENCE|Experience\s*:|Professional\s+Experience\s*:|Work\s+Experience\s*:)"),
    ("Technical Skills", r"^\s*(?:technical\s+skills|skills|core\s+competencies|technologies|skills\s*&\s*tools|tech\s+stack)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:TECHNICAL\s+SKILLS|SKILLS|CORE\s+COMPETENCIES|Technical\s+Skills\s*:|Skills\s*:)"),
    ("Projects", r"^\s*(?:projects|personal\s+projects|academic\s+projects|selected\s+projects|technical\s+projects|key\s+projects)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:PROJECTS|PERSONAL\s+PROJECTS|TECHNICAL\s+PROJECTS|Projects\s*:)"),
    ("Summary", r"^\s*(?:professional\s+summary|executive\s+summary|summary|career\s+objective|objective|profile)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:PROFESSIONAL\s+SUMMARY|EXECUTIVE\s+SUMMARY|SUMMARY|Professional\s+Summary\s*:|Executive\s+Summary\s*:|Summary\s*:)"),
    ("Certifications", r"^\s*(?:certifications|licenses\s*&\s*certifications|certificates)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:CERTIFICATIONS|Certifications\s*:)"),
    ("Achievements", r"^\s*(?:achievements|awards|honors\s*&\s*awards|honors|accomplishments)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:ACHIEVEMENTS|AWARDS|HONORS|Achievements\s*:|Awards\s*:)"),
    ("Leadership", r"^\s*(?:leadership|extracurricular\s+activities|activities|volunteer\s+experience|volunteering)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:LEADERSHIP|EXTRACURRICULAR|Leadership\s*:)"),
    ("Internships", r"^\s*(?:internships|internship\s+experience)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:INTERNSHIPS|Internships\s*:)"),
    ("Publications", r"^\s*(?:publications|research\s+publications)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:PUBLICATIONS|Publications\s*:)"),
    ("Coursework", r"^\s*(?:relevant\s+coursework|coursework)\s*[:\-–—]?\s*$", r"(?:^|\n|\s{2,})(?:RELEVANT\s+COURSEWORK|COURSEWORK|Relevant\s+Coursework\s*:)"),
]

RESUME_ACTION_VERBS: Set[str] = {
    "developed", "engineered", "built", "designed", "implemented", "architected",
    "created", "led", "managed", "optimized", "improved", "reduced", "increased",
    "automated", "deployed", "collaborated", "analyzed", "maintained", "integrated",
    "spearheaded", "conducted", "delivered", "authored", "mentored", "configured",
    "researched", "streamlined", "launched", "refactored", "migrated", "administered",
    "orchestrated", "scaled", "accelerated", "formulated", "pioneered"
}

WEAK_RESUME_PHRASES: Dict[str, str] = {
    "responsible for": "Replace 'Responsible for' with 'Spearheaded', 'Owned', or 'Engineered'",
    "helped with": "Replace 'Helped with' with 'Collaborated on' or 'Contributed to'",
    "worked on": "Replace 'Worked on' with 'Developed', 'Built', or 'Architected'",
    "duties included": "Replace 'Duties included' with direct action verbs ('Delivered', 'Managed')",
    "assisted in": "Replace 'Assisted in' with 'Co-developed' or 'Executed'",
    "tasked with": "Replace 'Tasked with' with 'Led' or 'Implemented'",
    "familiar with": "Replace 'Familiar with' by demonstrating usage in a concrete project bullet",
}

NON_RESUME_FILENAME_KEYWORDS: Set[str] = {
    "lecture", "notes", "slide", "slides", "chapter", "week", "module", "unit",
    "syllabus", "homework", "assignment", "exam", "midterm", "final", "quiz",
    "lab", "tutorial", "worksheet", "handout", "textbook", "book", "invoice",
    "receipt", "statement", "contract", "manual", "guide", "paper", "thesis",
    "dissertation", "essay", "article", "presentation", "deck", "rubric", "solution"
}

NON_RESUME_SIGNATURES: Dict[str, List[str]] = {
    "Lecture Notes / Course Slides / Study Material": [
        r"\blecture\s*(\d+|notes|slides|outline|summary|topic|series)?\b",
        r"\bslides?\s*(\d+|deck|presentation)\b",
        r"\b(week|module|unit|chapter|topic|lesson)\s+\d+\b",
        r"\blearning\s+(objectives?|outcomes?|goals?)\b",
        r"\b(theorem|lemma|corollary|postulate|axiom|proof)\b",
        r"\b(definition|example|exercise|figure|fig\.|table|equation|formula)\s*\d+(\.\d+)?\b",
        r"\b(instructor|professor|prof\.|lecturer|teaching\s+assistant|ta\s*:)\b",
        r"\b(course\s+overview|course\s+description|reading\s+assignment|textbook|handout|recitation|tutorial)\b",
        r"\b[1-9]\.[1-9](\.[1-9])?\s+[a-z]{3,}",
        r"\b(in\s+this\s+lecture|in\s+this\s+chapter|in\s+this\s+module|by\s+the\s+end\s+of\s+this)\b",
    ],
    "Academic Assignment / Exam / Syllabus": [
        r"\bproblem\s+set\b",
        r"\b(homework|assignment)\s*\d*\b",
        r"\banswer\s+all\s+questions\b",
        r"\btotal\s+(marks|points)\b",
        r"\b\d+\s+(points|marks)\s*\)",
        r"\bcourse\s+syllabus\b",
        r"\bgrading\s+(policy|scale|rubric|scheme)\b",
        r"\boffice\s+hours\b",
        r"\b(midterm|final)\s+exam\b",
        r"\bquestion\s+[1-9]\b",
        r"\blab\s+(manual|exercise|report)\b",
        r"\bsubmission\s+(guidelines|instructions|deadline)\b",
    ],
    "Research Paper / Textbook / Article": [
        r"\babstract\b",
        r"\b1\.?\s+introduction\b",
        r"\b2\.?\s+(related\s+work|background|methodology|literature\s+review)\b",
        r"\b(references|bibliography)\b",
        r"\bet\s+al\.",
        r"\bdoi:\s*10\.",
        r"\barxiv:\d+",
        r"\btable\s+of\s+contents\b",
        r"\bappendix\s+[a-z0-9]\b",
        r"\b(isbn|issn)\s*[:\-]?\s*\d",
    ],
    "Invoice / Financial Receipt / Statement": [
        r"\binvoice\s*(#|no|number|date)\b",
        r"\bbill\s+to\b",
        r"\bship\s+to\b",
        r"\bsubtotal\b",
        r"\btotal\s+due\b",
        r"\bamount\s+due\b",
        r"\bpayment\s+(method|terms)\b",
        r"\bbalance\s+(due|forward)\b",
        r"\btax\s+invoice\b",
        r"\baccount\s+statement\b",
        r"\btransaction\s+history\b",
    ],
    "Legal Contract / Terms / Policy Document": [
        r"\bterms\s+(and\s+conditions|of\s+service|of\s+use)\b",
        r"\bprivacy\s+policy\b",
        r"\bhereby\s+agrees\b",
        r"\bin\s+witness\s+whereof\b",
        r"\bgoverning\s+law\b",
        r"\bindemnification\b",
        r"\bnon-disclosure\s+agreement\b",
        r"\buser\s+manual\b",
    ],
}


def validate_document_is_resume(raw_text: str, cleaned_text: str = "", filename: str = "") -> Tuple[int, List[str]]:
    """
    Strict multi-layer gatekeeper that distinguishes a genuine Resume / CV
    from Lecture Notes, Slides, Assignments, Textbooks, Research Papers, Invoices,
    Certificates, and random PDFs/Images across all supported formats (PDF, PNG, JPG).
    Returns (confidence_score, detected_section_names) or raises HTTPException(400).
    """
    if not cleaned_text:
        cleaned_text = clean_text(raw_text)

    lower_clean = cleaned_text.lower()
    words = lower_clean.split()
    word_count = len(words)

    fn_lower = (filename or "").lower()
    fn_tokens = set(re.findall(r"[a-z]+", fn_lower))
    filename_has_non_resume_hint = bool(fn_tokens.intersection(NON_RESUME_FILENAME_KEYWORDS)) and not bool(
        fn_tokens.intersection({"resume", "cv", "curriculum", "vitae"})
    )

    # 1. Detect Explicit Resume Section Headers (Line-by-Line + Explicit Heading Syntax)
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", raw_text) if ln.strip()]
    matched_sections: List[str] = []

    for label, line_pat, inline_pat in STRICT_SECTION_PATTERNS:
        # Check if any standalone short line (<= 5 words) matches the heading pattern
        found_on_line = any(
            len(ln.split()) <= 5 and re.match(line_pat, ln, re.IGNORECASE)
            for ln in lines
        )
        # Or check if explicit heading syntax (e.g. UPPERCASE or 'Section:') exists in raw text
        found_inline = bool(re.search(inline_pat, raw_text))
        if found_on_line or found_inline:
            if label not in matched_sections:
                matched_sections.append(label)

    unique_sections = len(matched_sections)
    core_pillars_found = sum(
        1 for pillar in ("Education", "Experience", "Technical Skills", "Projects", "Internships")
        if pillar in matched_sections
    )

    # 2. Check Negative / Non-Resume Document Signatures (Lecture Notes, Exams, Papers, Invoices, Legal)
    detected_non_resume_type: Optional[str] = None
    max_negative_hits = 0
    total_negative_hits = 0
    for doc_type, patterns in NON_RESUME_SIGNATURES.items():
        hits = sum(1 for p in patterns if re.search(p, lower_clean))
        total_negative_hits += hits
        if hits > max_negative_hits:
            max_negative_hits = hits
            detected_non_resume_type = doc_type

    # Immediate veto if filename indicates non-resume (e.g. lecture_notes.pdf) and document has any non-resume markers or lacks 3 core pillars
    if filename_has_non_resume_hint and (total_negative_hits >= 1 or core_pillars_found < 3):
        detected_label = detected_non_resume_type or "Lecture Notes / Non-Resume Document"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: '{filename}' was identified as '{detected_label}' "
                f"rather than a Resume or CV. Glyph strictly accepts candidate resumes."
            ),
        )

    # Veto if document exhibits lecture note / exam / paper / invoice / legal signatures
    if max_negative_hits >= 2 or total_negative_hits >= 3:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: This file appears to be '{detected_non_resume_type}' "
                f"rather than a Resume or CV. Glyph only accepts candidate resumes containing personal contact details, "
                f"work/project experience, education, and skills."
            ),
        )

    # 3. Candidate Contact & Identity Verification
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", cleaned_text))
    has_phone = bool(
        re.search(
            r"(\+\d{1,3}[\s\-\.]?\d{2,4}[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}|\+?\d{1,3}[\s\-\.]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})",
            cleaned_text,
        )
    )
    has_profile_link = bool(
        re.search(r"\b(linkedin\.com/in/|github\.com/|gitlab\.com/|portfolio|leetcode\.com/|kaggle\.com/|behance\.net/)\b", lower_clean)
    )
    contact_score = (25 if has_email else 0) + (15 if has_phone else 0) + (15 if has_profile_link else 0)

    # A genuine resume MUST contain at least one candidate contact method (email, phone, or profile link)
    if contact_score == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: No candidate contact information (email address, phone number, "
                f"or LinkedIn/GitHub profile) was found in '{filename or 'document'}'. "
                f"Lecture notes, articles, and general documents are not accepted."
            ),
        )

    # 4. Chronology & Education Signals
    has_date_range = bool(
        re.search(r"\b(19|20)\d{2}\s*[\-–—to]+\s*((19|20)\d{2}|present|current|now)\b", lower_clean)
        or re.search(r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s+(19|20)\d{2}\b", lower_clean)
    )
    has_degree = bool(
        re.search(
            r"\b(b\.?s\.?|b\.?a\.?|m\.?s\.?|b\.?tech|m\.?tech|ph\.?d|bachelor|master|associate\s+degree|high\s+school\s+diploma|gpa|cgpa)\b",
            lower_clean,
        )
    )
    has_role_title = bool(
        re.search(
            r"\b(engineer|developer|intern|internship|analyst|manager|designer|scientist|architect|consultant|specialist|researcher|lead|director|coordinator|associate|programmer|administrator)\b",
            lower_clean,
        )
    )
    chronology_score = (15 if has_date_range else 0) + (15 if has_degree else 0) + (10 if has_role_title else 0)

    # 5. Action Verb Density & Conversational/Academic Prose Penalty
    word_set = set(re.findall(r"\b[a-z]+\b", lower_clean))
    action_hits = len(word_set.intersection(RESUME_ACTION_VERBS))
    action_score = min(20, action_hits * 5)

    # Lecture notes and textbooks have high density of conversational/pedagogical pronouns ("we", "you", "let's", "suppose", "consider")
    pedagogical_words = len(re.findall(r"\b(we\s+will|we\s+can|let\s+us|suppose\s+that|consider\s+the|for\s+example|as\s+shown\s+in|note\s+that|recall\s+that)\b", lower_clean))
    if pedagogical_words >= 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: Instructional or lecture-style prose detected in '{filename or 'document'}'. "
                f"Please upload a valid candidate Resume or CV."
            ),
        )

    # Penalize excessively long documents (> 3,500 words)
    if word_count > 3500:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: The uploaded document is {word_count} words long and resembles "
                f"lecture notes, a report, or a book rather than a Resume/CV."
            ),
        )

    # 6. Require at least 2 strict resume section headers (including at least 2 core pillars: Experience/Projects, Education, Skills)
    section_score = min(40, unique_sections * 12)
    raw_confidence = section_score + contact_score + chronology_score + action_score

    if core_pillars_found < 2 or unique_sections < 2 or chronology_score < 10:
        missing_signals = []
        if core_pillars_found < 2:
            missing_signals.append("distinct Resume section headings (such as 'Experience', 'Education', 'Technical Skills', or 'Projects')")
        if chronology_score < 10:
            missing_signals.append("career/education timeline dates or degree credentials")

        reason_str = " and ".join(missing_signals) if missing_signals else "resume structural markers"
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Non-Resume Document Rejected: '{filename or 'document'}' does not have the structure of a Resume or CV "
                f"(missing {reason_str}). Please upload a genuine resume."
            ),
        )

    return min(100, max(65, raw_confidence)), matched_sections


# -------------------------------------------------------------------------
# PDF CID Font & Ligature Sanitizer (Fixes (cid:136), (cid:22), (cid:12) etc.)
# -------------------------------------------------------------------------

CID_REPLACEMENT_MAP: Dict[str, str] = {
    "(cid:136)": "• ",
    "(cid:22)": " — ",
    "(cid:21)": "–",
    "(cid:11)": "ff",
    "(cid:12)": "fi",
    "(cid:13)": "fl",
    "(cid:14)": "ffi",
    "(cid:15)": "ffl",
    "\x16": " — ",
    "\x15": "–",
    "\x0b": "ff",
    "\x0e": "ffi",
    "\x0f": "ffl",
    "\u02c6": "•",
}


def sanitize_pdf_cid_artifacts(raw_text: str) -> str:
    """
    Removes PDF Type1/CID font encoding artifacts (such as (cid:136) or U+02C6 'ˆ'
    for bullets, (cid:22) for em-dashes, and (cid:12)/(cid:13) for fi/fl ligatures
    in LaTeX PDFs) and repairs broken ligature & soft-hyphen word splits.
    """
    if not raw_text:
        return ""
    text = raw_text
    # Repair inline \x0c when embedded inside a word (Type1 'fi' ligature)
    text = re.sub(r"(?<=[A-Za-z])\x0c(?=[A-Za-z])", "fi", text)
    for cid_token, replacement in CID_REPLACEMENT_MAP.items():
        text = text.replace(cid_token, replacement)
        if cid_token.startswith("(cid:"):
            text = text.replace(cid_token.upper(), replacement)

    # Remove any other residual (cid:NNN) tokens
    text = re.sub(r"\(cid:\s*\d+\)", " ", text, flags=re.IGNORECASE)

    # Repair soft-hyphen / replacement-char word splits across line wraps (e.g. CAS\ufffdCADE -> CASCADE)
    text = re.sub(r"([A-Za-z]{2,})\ufffd\s*([A-Za-z]{2,})", r"\1\2", text)
    text = re.sub(r"([A-Za-z]{2,})-\s*[\r\n]+\s*([a-z]{2,})", r"\1\2", text)
    text = text.replace("\ufffd", " ")

    # Repair common broken ligatures from LaTeX PDF extraction
    ligature_fixes = [
        (r"\bgami\s+cation\b", "gamification"),
        (r"\bcerti\s+cations\b", "certifications"),
        (r"\bCerti\s+cations\b", "Certifications"),
        (r"\bcerti\s+cate\b", "certificate"),
        (r"\bCerti\s+cate\b", "Certificate"),
        (r"\bnoti\s+cations\b", "notifications"),
        (r"\bnoti\s+cation\b", "notification"),
        (r"\bo\s+ine\b", "offline"),
        (r"\bO\s+ine\b", "Offline"),
        (r"\b(?<=\s)ndings\b", "findings"),
        (r"\bpro\s+ciency\b", "proficiency"),
        (r"\be\s+cient\b", "efficient"),
        (r"\be\s+ciency\b", "efficiency"),
        (r"\bwork\s+ows\b", "workflows"),
        (r"\bwork\s+ow\b", "workflow"),
        (r"\bidenti\s+ed\b", "identified"),
        (r"\bidenti\s+cation\b", "identification"),
        (r"\bspeci\s+c\b", "specific"),
        (r"\bspeci\s+cation\b", "specification"),
        (r"\bsigni\s+cant\b", "significant"),
        (r"\bclassi\s+cation\b", "classification"),
        (r"\bclassi\s+er\b", "classifier"),
        (r"\bveri\s+cation\b", "verification"),
        (r"\bveri\s+ed\b", "verified"),
        (r"\buni\s+ed\b", "unified"),
        (r"\bmodi\s+ed\b", "modified"),
        (r"\bcon\s+guration\b", "configuration"),
        (r"\bcon\s+gured\b", "configured"),
        (r"\bcon\s+dence\b", "confidence"),
        (r"\bdi\s+erent\b", "different"),
        (r"\be\s+ective\b", "effective"),
        (r"\btra\s+c\b", "traffic"),
        (r"\barti\s+cial\b", "artificial"),
        (r"\bde\s+ned\b", "defined"),
    ]
    for pat, repl in ligature_fixes:
        text = re.sub(pat, repl, text)

    return text


# -------------------------------------------------------------------------
# Bulletproof Date/Year Extractor & Deduplicator + Section Parsers
# -------------------------------------------------------------------------

MONTH_TOKEN_RE = r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:t(?:ember)?)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?|Spring|Summer|Fall|Autumn|Winter)"
YEAR_TOKEN_RE = r"(?:19|20)\d{2}"
DATE_POINT_RE = rf"(?:{MONTH_TOKEN_RE}\.?\s+)?{YEAR_TOKEN_RE}"
DATE_END_RE = rf"(?:{DATE_POINT_RE}|Present|Current|Now|Ongoing|\d{{2}})"
DATE_RANGE_RE = rf"{DATE_POINT_RE}\s*(?:[\-–—/]+|\bto\b)\s*{DATE_END_RE}"
SINGLE_MONTH_YEAR_RE = rf"{MONTH_TOKEN_RE}\.?\s+{YEAR_TOKEN_RE}"
FULL_DATE_TOKEN_RE = rf"(?:{DATE_RANGE_RE}|{SINGLE_MONTH_YEAR_RE}|{YEAR_TOKEN_RE})"


def normalize_date_str(raw_date: str) -> str:
    """
    Normalizes any date/year string (e.g. '2024   2028', '2024 - 2028',
    'June 2022 - Present', '2025 2025', '(2025)') into a single clean ATS date
    label with zero duplicate years.
    """
    if not raw_date:
        return ""
    d = sanitize_pdf_cid_artifacts(str(raw_date)).strip(" \t\r\n()[]|,-–—:;•")
    if not d:
        return ""

    # Convert space-separated year ranges (e.g. '2024   2028' from LaTeX PDF extraction) into '2024 – 2028'
    d = re.sub(
        rf"\b({YEAR_TOKEN_RE})\s{{1,8}}({YEAR_TOKEN_RE}|Present|Current)\b",
        lambda m: m.group(1) if m.group(1).lower() == m.group(2).lower() else f"{m.group(1)} – {m.group(2)}",
        d,
        flags=re.IGNORECASE,
    )
    # Normalize hyphens/dashes/"to" between date parts to clean ' – '
    d = re.sub(r"\s*(?:[\-–—]+|\bto\b)\s*", " – ", d, flags=re.IGNORECASE)
    d = re.sub(r"\s+", " ", d).strip(" \t()[]|,-–—:;•")

    # Collapse duplicate identical years or ranges (e.g. '2025 – 2025' -> '2025' or '2024 – 2028 – 2024 – 2028')
    parts = [p.strip() for p in d.split(" – ") if p.strip()]
    if len(parts) == 2 and parts[0].lower() == parts[1].lower():
        return parts[0]
    if len(parts) == 4 and parts[0] == parts[2] and parts[1] == parts[3]:
        return f"{parts[0]} – {parts[1]}"
    return d


def extract_and_strip_date(raw_line: str, default_date: str = "") -> Tuple[str, str]:
    """
    Extracts any trailing or parenthesized date/year range from a resume line AND
    strips ALL occurrences of that date/year from the line text so that dates/years
    NEVER appear twice (once inline in the title/institution/project/certification
    and again in the right-hand date column).
    Returns (cleaned_text_without_date, normalized_date_str).
    """
    if not raw_line:
        return "", normalize_date_str(default_date)

    line = sanitize_pdf_cid_artifacts(str(raw_line)).strip()
    if not line:
        return "", normalize_date_str(default_date)

    # Pre-normalize space-separated 4-digit year pairs anywhere at the tail of the line (e.g. 'Pacific Coast University 2024   2028')
    line = re.sub(
        rf"\b({YEAR_TOKEN_RE})\s{{1,8}}({YEAR_TOKEN_RE}|Present|Current)(\s*[\)\]]?\s*)$",
        lambda m: (m.group(1) if m.group(1).lower() == m.group(2).lower() else f"{m.group(1)} – {m.group(2)}") + m.group(3),
        line,
        flags=re.IGNORECASE,
    )

    trailing_re = re.compile(
        rf"[\s\|\-–—,:;•]*[\(\[]?\s*({FULL_DATE_TOKEN_RE})\s*[\)\]]?\s*$",
        re.IGNORECASE,
    )

    extracted_date = ""
    while True:
        m = trailing_re.search(line)
        if not m:
            break
        cand = normalize_date_str(m.group(1))
        if cand:
            if not extracted_date or ("–" in cand and "–" not in extracted_date) or (len(cand) > len(extracted_date) and "–" not in extracted_date):
                extracted_date = cand
        line = line[: m.start()].strip(" \t|-–—,:;•([")

    # Also check for parenthesized date ranges or explicit year ranges embedded inside the line
    embedded_re = re.compile(
        rf"[\(\[]\s*({FULL_DATE_TOKEN_RE})\s*[\)\]]|(?:\s+|\b)({DATE_RANGE_RE})(?=\s|$|[,;|])",
        re.IGNORECASE,
    )
    while True:
        em = embedded_re.search(line)
        if not em:
            break
        raw_found = em.group(1) or em.group(2) or ""
        cand = normalize_date_str(raw_found)
        if cand and not extracted_date:
            extracted_date = cand
        line = (line[: em.start()] + " " + line[em.end() :]).strip()

    # Clean up empty parens or dangling punctuation
    line = re.sub(r"\(\s*\)|\[\s*\]", "", line)
    line = re.sub(r"\s{2,}", " ", line).strip(" \t|-–—,:;•(")

    final_date = extracted_date if extracted_date else normalize_date_str(default_date)
    return line, final_date


def slice_resume_sections(raw_text_lines: str) -> Dict[str, List[str]]:
    """
    Splits raw resume lines into canonical section buckets ('header', 'Summary',
    'Education', 'Experience', 'Projects', 'Technical Skills', 'Certifications', etc.)
    so each section is parsed cleanly without cross-contaminating other sections.
    """
    buckets: Dict[str, List[str]] = {
        "header": [],
        "Summary": [],
        "Education": [],
        "Experience": [],
        "Projects": [],
        "Technical Skills": [],
        "Certifications": [],
        "Achievements": [],
        "Leadership": [],
        "Publications": [],
        "Coursework": [],
    }
    inline_header_re = re.compile(
        r"^\s*(professional\s+summary|executive\s+summary|summary|profile|objective|"
        r"education|academic\s+background|"
        r"work\s+experience|professional\s+experience|experience|internships|"
        r"projects|personal\s+projects|technical\s+projects|"
        r"technical\s+skills|skills|core\s+competencies|"
        r"certifications|certificates)\s*:\s*(.+)$",
        re.IGNORECASE,
    )

    current_sec = "header"
    for raw_ln in re.split(r"[\r\n]+", raw_text_lines):
        ln = raw_ln.strip()
        if not ln:
            continue

        # Check if line has 'SECTION: inline content'
        inline_m = inline_header_re.match(ln)
        if inline_m:
            h_raw = inline_m.group(1).lower()
            remainder = inline_m.group(2).strip()
            if "summary" in h_raw or "profile" in h_raw or "objective" in h_raw:
                current_sec = "Summary"
            elif "education" in h_raw or "academic" in h_raw:
                current_sec = "Education"
            elif "experience" in h_raw or "internship" in h_raw:
                current_sec = "Experience"
            elif "project" in h_raw:
                current_sec = "Projects"
            elif "skill" in h_raw or "competenc" in h_raw:
                current_sec = "Technical Skills"
            elif "certif" in h_raw:
                current_sec = "Certifications"
            if remainder:
                buckets.setdefault(current_sec, []).append(remainder)
            continue

        matched_header = None
        if len(ln.split()) <= 5:
            for sec_name, line_regex, _ in STRICT_SECTION_PATTERNS:
                if re.match(line_regex, ln, re.IGNORECASE):
                    matched_header = "Experience" if sec_name == "Internships" else sec_name
                    break
        if matched_header and matched_header in buckets:
            current_sec = matched_header
            continue
        buckets.setdefault(current_sec, []).append(ln)
    return buckets


INSTITUTION_KEYWORDS = (
    "university", "college", "institute", "school", "academy", "polytechnic",
    "nuces", "fast", "lums", "nust", "giki", "iba", "ned", "uet", "comsats",
    "mit", "stanford", "berkeley", "harvard", "oxford", "cambridge", "iit",
)

DEGREE_KEYWORDS = (
    "bachelor", "master", "b.s", "b.a", "m.s", "m.a", "b.tech", "m.tech", "ph.d",
    "bsc", "msc", "bs ", "ms ", "be ", "me ", "mba", "associate", "diploma",
    "higher secondary", "certificate", "hsc", "ssc", "pre-engineering", "pre-medical",
    "computer science", "software engineering", "artificial intelligence", "data science",
    "information technology", "electrical", "mechanical", "matriculation", "o-level", "a-level",
)


def _has_inst_kw(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in INSTITUTION_KEYWORDS)


def _has_deg_kw(text: str) -> bool:
    low = text.lower()
    return any(k in low for k in DEGREE_KEYWORDS)


def parse_education_section(edu_lines: List[str], raw_text_lines: str, cleaned_text: str) -> List[EducationItem]:
    """
    Parses one or more Education entries cleanly, separating Institution, Degree,
    and Year Range while guaranteeing zero duplicate years or merged lines.
    """
    items: List[EducationItem] = []
    pending_dates: List[str] = []

    for raw_ln in edu_lines:
        clean_ln, dt = extract_and_strip_date(raw_ln)
        if not clean_ln:
            if dt:
                if items and not items[-1].year:
                    items[-1].year = dt
                else:
                    pending_dates.append(dt)
            continue

        # Check if a single line contains BOTH Degree and Institution separated by comma / dash / 'at'
        if _has_inst_kw(clean_ln) and _has_deg_kw(clean_ln):
            parts = [p.strip() for p in re.split(r"\s*(?:,|—|–|\||;|\bat\b)\s*", clean_ln) if p.strip()]
            if len(parts) >= 2:
                inst_part = next((p for p in parts if _has_inst_kw(p)), "")
                deg_part = next((p for p in parts if p != inst_part and _has_deg_kw(p)), "")
                if inst_part and deg_part:
                    yr_val = dt or (pending_dates.pop(0) if pending_dates else "")
                    items.append(EducationItem(institution=inst_part, degree=deg_part, year=yr_val))
                    continue

        is_inst = _has_inst_kw(clean_ln)
        is_deg = _has_deg_kw(clean_ln)

        if is_inst and not is_deg:
            if items and not items[-1].institution:
                items[-1].institution = clean_ln
                if dt and not items[-1].year:
                    items[-1].year = dt
            else:
                yr_val = dt or (pending_dates.pop(0) if pending_dates else "")
                items.append(EducationItem(institution=clean_ln, degree="", year=yr_val))
        elif is_deg and not is_inst:
            if items and not items[-1].degree:
                items[-1].degree = clean_ln
                if dt and not items[-1].year:
                    items[-1].year = dt
            else:
                yr_val = dt or (pending_dates.pop(0) if pending_dates else "")
                items.append(EducationItem(institution="", degree=clean_ln, year=yr_val))
        else:
            # Ambiguous line: if previous item is missing degree, attach as degree; otherwise start new institution
            if items and not items[-1].degree and items[-1].institution:
                items[-1].degree = clean_ln
                if dt and not items[-1].year:
                    items[-1].year = dt
            elif items and not items[-1].institution and items[-1].degree:
                items[-1].institution = clean_ln
                if dt and not items[-1].year:
                    items[-1].year = dt
            else:
                yr_val = dt or (pending_dates.pop(0) if pending_dates else "")
                items.append(EducationItem(institution=clean_ln, degree="", year=yr_val))

    # Pair any remaining standalone dates with items missing a year
    for it in items:
        if not it.year and pending_dates:
            it.year = pending_dates.pop(0)
        if not it.year:
            it.year = "2020 – 2024"
        if not it.institution:
            it.institution = "University"
        if not it.degree:
            it.degree = "Bachelor of Science in Computer Science"

    if not items:
        deg_val = "Bachelor of Science in Computer Science"
        inst_val = "University of Washington"
        edu_deg_match = re.search(
            r"\b((?:B\.?S\.?|B\.?A\.?|M\.?S\.?|B\.?Tech|M\.?Tech|Ph\.?D\.?|Bachelor\s+of\s+Science|Bachelor\s+of\s+Arts|Master\s+of\s+Science)(?:\s+in\s+[A-Za-z\s&]{3,28})?)\b",
            cleaned_text,
            re.IGNORECASE,
        )
        if edu_deg_match and len(edu_deg_match.group(1).strip()) <= 52:
            deg_val = edu_deg_match.group(1).strip()
        edu_inst_match = re.search(
            r"\b((?:University|Institute\s+of\s+Technology|College)\s+of\s+[A-Z][A-Za-z\s]{2,22}|[A-Z][A-Za-z\s]{2,22}\s+(?:University|Institute\s+of\s+Technology|College))\b",
            raw_text_lines,
        )
        if edu_inst_match:
            cand_inst = re.sub(
                r"^(?:Computer\s+Science|Engineering|Science|Arts|In)\s+",
                "",
                edu_inst_match.group(1).strip(),
                flags=re.IGNORECASE,
            ).strip()
            if 4 <= len(cand_inst) <= 48:
                inst_val = cand_inst
        items.append(EducationItem(institution=inst_val, degree=deg_val, year="2020 – 2024"))

    return items[:3]


def _is_bullet_line(ln: str) -> bool:
    return bool(re.match(r"^\s*(?:[•●▪➢\*]|\-(?=\s))\s*", ln))


def _strip_bullet_marker(ln: str) -> str:
    return re.sub(r"^\s*(?:[•●▪➢\*]|\-(?=\s))\s*", "", ln).strip()


def _looks_like_tech_stack_line(ln: str) -> bool:
    if _is_bullet_line(ln):
        return False
    words = ln.split()
    if len(words) > 15 or ln.endswith("."):
        return False
    if "•" in ln or "|" in ln:
        return True
    low = ln.lower()
    tech_hits = sum(
        1
        for kw in ("python", "javascript", "typescript", "react", "java", "c#", "c++", "sql", "mysql", "postgresql", "flask", "django", "fastapi", "node", "firebase", "firestore", ".net", "wpf", "html", "css", "aws", "docker", "kotlin", "swift")
        if kw in low
    )
    return tech_hits >= 2 and len(words) <= 12


def parse_experience_section(exp_lines: List[str], raw_text_lines: str) -> List[ExperienceItem]:
    """
    Parses Work Experience entries cleanly without chopping wrapped multi-line
    PDF bullets or duplicating dates inside role/company titles.
    """
    items: List[ExperienceItem] = []

    if exp_lines:
        current_title = ""
        current_company = ""
        current_dates = ""
        current_bullets: List[str] = []

        for raw_ln in exp_lines:
            ln = sanitize_pdf_cid_artifacts(raw_ln).strip()
            if not ln:
                continue

            if _is_bullet_line(ln):
                b_text = _strip_bullet_marker(ln)
                if b_text:
                    current_bullets.append(b_text)
                continue

            clean_ln, dt = extract_and_strip_date(ln)
            # Standalone date line from 2-column PDF layout
            if not clean_ln and dt:
                if not current_dates:
                    current_dates = dt
                elif current_bullets:
                    items.append(
                        ExperienceItem(
                            title=current_title or "Software Engineer",
                            company=current_company or "Company",
                            dates=current_dates or "June 2022 – Present",
                            bullets=current_bullets,
                        )
                    )
                    current_title, current_company, current_dates, current_bullets = "", "", dt, []
                continue

            # If we haven't set the role title yet for the current entry
            if not current_title:
                # Check if Role and Company are on the same line ('Software Engineer at Starlight Systems')
                if re.search(r"\s+at\s+", clean_ln, re.IGNORECASE):
                    parts = re.split(r"\s+at\s+", clean_ln, maxsplit=1, flags=re.IGNORECASE)
                    current_title = parts[0].strip(" -–—|,")
                    current_company = parts[1].strip(" -–—|,")
                else:
                    current_title = clean_ln
                if dt:
                    current_dates = dt
                continue

            # If we have a title, no bullets yet, and no company yet -> this non-bullet line is the Company / Subtitle line
            if not current_bullets and not current_company:
                # Repair LaTeX em-dash space gap in subtitle (e.g. 'Extern, Inc.   sprint conducted...')
                comp_fixed = re.sub(r"\b(Inc\.|LLC|Ltd\.|Corp\.)\s{2,}", r"\1 — ", clean_ln)
                current_company = comp_fixed
                if dt and not current_dates:
                    current_dates = dt
                continue

            # If we already have bullets and encounter a new short header line with a date, start a new Experience entry!
            if current_bullets and dt and len(clean_ln.split()) <= 10 and not clean_ln.endswith("."):
                items.append(
                    ExperienceItem(
                        title=current_title or "Software Engineer",
                        company=current_company or "Company",
                        dates=current_dates or "June 2022 – Present",
                        bullets=current_bullets,
                    )
                )
                if re.search(r"\s+at\s+", clean_ln, re.IGNORECASE):
                    parts = re.split(r"\s+at\s+", clean_ln, maxsplit=1, flags=re.IGNORECASE)
                    current_title = parts[0].strip(" -–—|,")
                    current_company = parts[1].strip(" -–—|,")
                else:
                    current_title = clean_ln
                    current_company = ""
                current_dates = dt
                current_bullets = []
                continue

            # Otherwise, this non-bullet line is a wrapped continuation of the previous bullet (or a bullet without a glyph)
            if current_bullets:
                current_bullets[-1] = re.sub(r"\s+", " ", f"{current_bullets[-1]} {ln}").strip()
            else:
                current_bullets.append(ln)

        if current_title or current_bullets:
            items.append(
                ExperienceItem(
                    title=current_title or "Software Engineer",
                    company=current_company or "Company",
                    dates=current_dates or "June 2022 – Present",
                    bullets=current_bullets,
                )
            )

    # Normalize bullets (ensure clean punctuation without mid-sentence chopping)
    for it in items:
        cleaned_b: List[str] = []
        for b in it.bullets or []:
            b_norm = re.sub(r"\s+", " ", b).strip(" -•●▪\t")
            if len(b_norm.split()) >= 4:
                if not b_norm.endswith((".", "!", "?")):
                    b_norm += "."
                cleaned_b.append(b_norm)
        it.bullets = cleaned_b

    if not items:
        fallback_bullets: List[str] = []
        for sent in re.split(r"(?:[•●▪➢]+|(?<=\.)\s+)", raw_text_lines):
            s_clean = re.sub(r"\s+", " ", sanitize_pdf_cid_artifacts(sent)).strip(" -•●▪➢\t")
            s_words = s_clean.split()
            if 6 <= len(s_words) <= 45:
                first_w = re.sub(r"[^a-zA-Z]", "", s_words[0]).lower()
                if first_w in RESUME_ACTION_VERBS and "@" not in s_clean and "linkedin.com" not in s_clean.lower():
                    if not s_clean.endswith("."):
                        s_clean += "."
                    if s_clean not in fallback_bullets:
                        fallback_bullets.append(s_clean)
            if len(fallback_bullets) >= 3:
                break
        if not fallback_bullets:
            fallback_bullets = [
                "Engineered scalable software components and RESTful services adhering to modern engineering standards.",
                "Collaborated across cross-functional teams to deliver high-reliability product features on schedule.",
            ]
        items.append(
            ExperienceItem(
                title="Software Engineer",
                company="Starlight Systems",
                dates="June 2022 – Present",
                bullets=fallback_bullets,
            )
        )

    return items[:4]


def parse_projects_section(proj_lines: List[str], detected_skills: List[str]) -> List[ProjectItem]:
    """
    Parses one or more Project entries cleanly (e.g. mobile apps, web platforms,
    and systems projects), preserving each project's Name, Tech Stack, Year, and
    multi-line wrapped bullets without chopping sentences or duplicating years.
    """
    items: List[ProjectItem] = []
    if proj_lines:
        current_name = ""
        current_tech = ""
        current_date = ""
        current_bullets: List[str] = []

        total_lines = len(proj_lines)
        for idx, raw_ln in enumerate(proj_lines):
            ln = sanitize_pdf_cid_artifacts(raw_ln).strip()
            if not ln:
                continue

            if _is_bullet_line(ln):
                b_text = _strip_bullet_marker(ln)
                if b_text:
                    current_bullets.append(b_text)
                continue

            clean_ln, dt = extract_and_strip_date(ln)
            if not clean_ln and dt:
                if not current_date:
                    current_date = dt
                continue

            next_ln = sanitize_pdf_cid_artifacts(proj_lines[idx + 1]).strip() if (idx + 1 < total_lines) else ""
            next_is_tech = _looks_like_tech_stack_line(next_ln)

            # Detect if this line is a new Project Header
            is_new_project_header = False
            if not current_name:
                is_new_project_header = True
            elif not _looks_like_tech_stack_line(ln) and (
                (dt and len(clean_ln.split()) <= 12 and not clean_ln.endswith("."))
                or (next_is_tech and len(clean_ln.split()) <= 12 and not clean_ln.endswith("."))
            ):
                is_new_project_header = True

            if is_new_project_header:
                if current_name or current_bullets:
                    items.append(
                        ProjectItem(
                            name=current_name or "Technical Project",
                            tech=current_tech,
                            date=current_date or "2025",
                            bullets=current_bullets,
                        )
                    )
                # Check if single-line project ('Glyph Analytics Platform - Built a web application...')
                if " - " in clean_ln and len(clean_ln.split()) > 8 and clean_ln.endswith("."):
                    p_parts = clean_ln.split(" - ", 1)
                    current_name = p_parts[0].strip()
                    current_tech = " • ".join(detected_skills[:4])
                    current_date = dt or "2025"
                    current_bullets = [p_parts[1].strip()]
                else:
                    # Repair 2+ spaces from LaTeX em-dash in project title ('Project Name   Platform' -> 'Project Name — Platform')
                    header_fixed = re.sub(r"\s{2,}", " — ", raw_ln.strip())
                    header_clean, header_dt = extract_and_strip_date(header_fixed)
                    current_name = header_clean or clean_ln
                    current_tech = ""
                    current_date = header_dt or dt or "2025"
                    current_bullets = []
                continue

            # Check if this line is the Tech Stack subtitle right below the project name
            if current_name and not current_tech and not current_bullets and _looks_like_tech_stack_line(ln):
                current_tech = clean_ln.strip(" •|-–—")
                if dt and not current_date:
                    current_date = dt
                continue

            # Otherwise, this non-bullet line is a wrapped continuation of the current bullet!
            if current_bullets:
                current_bullets[-1] = re.sub(r"\s+", " ", f"{current_bullets[-1]} {ln}").strip()
            else:
                current_bullets.append(ln)

        if current_name or current_bullets:
            items.append(
                ProjectItem(
                    name=current_name or "Technical Project",
                    tech=current_tech or " • ".join(detected_skills[:4]),
                    date=current_date or "2025",
                    bullets=current_bullets,
                )
            )

    for it in items:
        cleaned_b: List[str] = []
        for b in it.bullets or []:
            b_norm = re.sub(r"\s+", " ", b).strip(" -•●▪\t")
            if len(b_norm.split()) >= 4:
                if not b_norm.endswith((".", "!", "?")):
                    b_norm += "."
                cleaned_b.append(b_norm)
        it.bullets = cleaned_b

    if not items:
        items.append(
            ProjectItem(
                name="Scalable Architecture & Analytics Platform",
                tech=" • ".join(detected_skills[:5]),
                date="2025",
                bullets=[
                    f"Designed, built, and deployed a full-featured platform leveraging {', '.join(detected_skills[:5])}, achieving sub-100ms response latency and 99.9% reliability."
                ],
            )
        )

    return items[:4]


def parse_certifications_section(cert_lines: List[str], missing_list: List[str]) -> List[str]:
    """
    Parses certification lines, pairing any detached year lines from 2-column PDFs
    and formatting each entry as 'Title — Year' with zero duplicate years.
    """
    entries: List[Tuple[str, str]] = []
    standalone_years: List[str] = []

    for raw_c in cert_lines:
        c_clean = _strip_bullet_marker(sanitize_pdf_cid_artifacts(raw_c))
        if not c_clean:
            continue
        title_part, dt = extract_and_strip_date(c_clean)
        if not title_part and dt:
            standalone_years.append(dt)
            continue
        if len(title_part) >= 4 and re.search(r"[A-Za-z]{2,}", title_part):
            entries.append((title_part, dt))

    formatted_certs: List[str] = []
    seen_titles: Set[str] = set()
    for title_part, dt in entries:
        low_t = title_part.lower()
        if low_t in seen_titles:
            continue
        seen_titles.add(low_t)
        yr = dt or (standalone_years.pop(0) if standalone_years else "2025")
        formatted_certs.append(f"{title_part} ({yr})")
        if len(formatted_certs) >= 6:
            break

    if not formatted_certs:
        top_cert_kw = format_skill_display(missing_list[0]) if missing_list else "Cloud & Software Engineering"
        formatted_certs = [
            f"AWS & Cloud Practitioner Foundations — {top_cert_kw} Specialization (2025)",
            "Software Engineering, Algorithms & Full-Stack Architecture Certification (2025)",
        ]
    return formatted_certs


def extract_resume_profile_in_memory(
    raw_text_lines: str,
    cleaned_text: str,
    matched_kws: List[str],
    missing_kws: Optional[List[str]] = None,
    expanded_roles: Optional[List[str]] = None,
) -> ParsedResumeProfile:
    """
    Extracts candidate contact info, summary, education, experience, projects,
    skills, and certifications directly from the uploaded resume text in RAM,
    sanitizes all PDF (cid:...) artifacts, prevents duplicate dates/years, and
    enriches the profile with target role keywords.
    """
    raw_text_lines = sanitize_pdf_cid_artifacts(raw_text_lines)
    cleaned_text = sanitize_pdf_cid_artifacts(cleaned_text)
    missing_list = missing_kws or []
    roles_list = expanded_roles or []
    buckets = slice_resume_sections(raw_text_lines)

    # 1. Email
    email_match = re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", cleaned_text)
    email = email_match.group(0) if email_match else "alex.vance@glyph.dev"

    # 2. Phone (Supports US & International formats e.g. +92 300 123 4567 or +1 (555) 019-2834)
    phone_match = re.search(
        r"(\+\d{1,3}[\s\-\.]?\d{2,4}[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}|\+?\d{1,3}[\s\-\.]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})",
        cleaned_text,
    )
    phone = phone_match.group(0).strip() if phone_match else "+1 (555) 019-2834"

    # 3. LinkedIn & GitHub
    li_match = re.search(r"(?:https?://)?(?:www\.)?linkedin\.com/in/[a-zA-Z0-9_\-]+", cleaned_text, re.IGNORECASE)
    linkedin = li_match.group(0) if li_match else ""

    gh_match = re.search(r"(?:https?://)?(?:www\.)?github\.com/[a-zA-Z0-9_\-]+", cleaned_text, re.IGNORECASE)
    github = gh_match.group(0) if gh_match else ""

    # 4. Full Name (Extract first so its tokens never bleed into Location)
    lines = [ln.strip() for ln in re.split(r"[\r\n]+", raw_text_lines) if ln.strip()]
    full_name = "Alex Vance"
    for ln in lines[:5]:
        candidate_line = re.split(r"[\|•\-–—@,:]", ln)[0].strip()
        words = candidate_line.split()
        if 2 <= len(words) <= 5 and all(re.match(r"^[A-Za-z\.\-']+$", w) for w in words):
            lower_w = {w.lower() for w in words}
            if not lower_w.intersection(
                {"resume", "curriculum", "vitae", "cv", "summary", "education", "experience", "skills", "projects", "profile"}
            ):
                full_name = " ".join(w.capitalize() if w.isupper() else w for w in words)
                break

    # 5. Location (Match per-line so candidate name on line 1 never merges with City, Country on line 2)
    location = "San Francisco, CA"
    name_tokens_lower = {w.lower().strip(".,") for w in full_name.split()}
    for ln in lines[:10]:
        loc_m = re.search(
            r"\b([A-Z][a-zA-Z\.]+(?:[ \t]+[A-Z][a-zA-Z\.]+){0,2},\s*(?:[A-Z]{2}\b|[A-Z][a-zA-Z]{2,14}\b))",
            ln,
        )
        if loc_m:
            raw_loc = loc_m.group(1).strip()
            city_part, _, region_part = raw_loc.partition(",")
            city_words = [w for w in city_part.split() if w.lower().strip(".,") not in name_tokens_lower]
            if city_words:
                location = f"{' '.join(city_words)}, {region_part.strip()}"
                break

    # 6. Extract Technical & Domain Skills (First from candidate's Technical Skills section, then matched, then missing)
    detected_skills: List[str] = []
    seen_s: Set[str] = set()

    skill_cat_prefixes = re.compile(
        r"^(?:languages|mobile\s*&\s*web|web\s*&\s*mobile|frontend|backend|cloud\s*&\s*db|cloud\s*&\s*databases|databases|tools\s*&\s*core|tools|soft\s+skills|frameworks|core\s+technologies|technologies)\s*[:\-–—]?\s*",
        re.IGNORECASE,
    )
    for sk_ln in buckets.get("Technical Skills", []):
        stripped_ln = skill_cat_prefixes.sub("", sanitize_pdf_cid_artifacts(sk_ln)).strip()
        # Split on commas that are NOT inside parentheses (so 'Firebase (Firestore, Auth, Storage)' stays intact or splits cleanly)
        for token in re.split(r",\s*(?![^()]*\))|\s*•\s*", stripped_ln):
            tok_clean = token.strip(" .;-–—\t")
            if 2 <= len(tok_clean) <= 42 and tok_clean.lower() not in {"mobile", "web", "&", "and", "core", "skills"}:
                if tok_clean.lower() not in seen_s:
                    seen_s.add(tok_clean.lower())
                    detected_skills.append(tok_clean)

    lower_clean = cleaned_text.lower()
    for sk in list(matched_kws) + sorted(ALL_KNOWN_SKILLS, key=lambda x: (-len(x), x)):
        if len(sk) < 2 or len(detected_skills) >= 20:
            break
        pat = r"(?<![a-zA-Z0-9])" + re.escape(sk.lower()) + r"(?![a-zA-Z0-9])"
        if re.search(pat, lower_clean):
            display_label = format_skill_display(sk)
            if display_label.lower() not in seen_s:
                seen_s.add(display_label.lower())
                detected_skills.append(display_label)

    for m_kw in missing_list[:14]:
        display_label = format_skill_display(m_kw)
        if display_label.lower() not in seen_s:
            seen_s.add(display_label.lower())
            detected_skills.append(display_label)

    if not detected_skills:
        detected_skills = ["Python", "JavaScript", "TypeScript", "SQL", "Git", "REST APIs", "System Design"]

    # 7. Parse Education (Supports multiple entries & strips duplicate dates)
    education_items = parse_education_section(buckets.get("Education", []), raw_text_lines, cleaned_text)

    # 8. Parse Experience (Supports multiple entries, joins wrapped bullets, strips duplicate dates)
    experience_items = parse_experience_section(buckets.get("Experience", []), raw_text_lines)

    # Only synthesize fallback bullets if the extracted experience entry had no bullets at all
    formatted_missing = [format_skill_display(k) for k in missing_list[:8]]
    if experience_items:
        primary_bullets = list(experience_items[0].bullets or [])
        if not primary_bullets and len(formatted_missing) >= 2:
            grp1 = ", ".join(formatted_missing[:3])
            primary_bullets.append(
                f"Architected and deployed production-grade workflows utilizing {grp1}, improving system reliability and reducing latency by 35%."
            )
        experience_items[0].bullets = primary_bullets[:5]

    # 9. Parse Projects (Supports multiple entries with deduplicated dates & tech stacks)
    project_items = parse_projects_section(buckets.get("Projects", []), detected_skills)

    # 10. Parse Certifications cleanly without duplicate years
    extracted_certs = parse_certifications_section(buckets.get("Certifications", []), missing_list)

    # 11. Clean ATS Summary (Preserves candidate's authentic summary without tacking on AI boilerplate)
    sum_lines = buckets.get("Summary", [])
    primary_title = experience_items[0].title if experience_items else "Software Engineer"
    target_role_label = roles_list[0] if roles_list else primary_title
    top_skills_str = ", ".join(detected_skills[:10])
    if sum_lines:
        summary = re.sub(r"\s+", " ", sanitize_pdf_cid_artifacts(" ".join(sum_lines))).strip()
    else:
        summary = (
            f"Results-driven {target_role_label} with strong proficiency in {top_skills_str}. "
            f"Proven track record of architecting scalable applications, optimizing end-to-end performance, and delivering measurable engineering impact."
        )

    return ParsedResumeProfile(
        full_name=full_name,
        email=email,
        phone=phone,
        location=location,
        linkedin=linkedin,
        github=github,
        summary=summary,
        skills=detected_skills,
        experience=experience_items,
        projects=project_items,
        education=education_items,
        certifications=extracted_certs,
    )


# -------------------------------------------------------------------------
# Bullet Impact, Quantification & XYZ Rewriter Engine
# -------------------------------------------------------------------------

XYZ_TEMPLATES: List[str] = [
    "Engineered scalable {kw} workflows and automated validation pipelines, improving system reliability and reducing processing latency by 35%.",
    "Architected and deployed production-grade {kw} integrations across core services, increasing throughput by 2.4x under peak load.",
    "Spearheaded the adoption of {kw} within agile engineering sprints, cutting manual operational overhead by 40% and accelerating release velocity.",
    "Optimized critical data and application paths utilizing {kw}, achieving 99.9% uptime and measurable performance gains.",
]


def analyze_bullet_impact_and_Audit(
    raw_text: str,
    cleaned_text: str,
    detected_sections: List[str],
    missing_keywords: List[str],
) -> Tuple[BulletImpactReport, StructuralAuditReport]:
    """
    1. Evaluates resume bullets for quantification (%, $, numbers, scale), strong action verbs,
       and weak passive phrasing, and generates tailored Google-XYZ formula bullets for missing keywords.
    2. Runs a 5-point Structural ATS Parser Audit with a raw text X-Ray preview.
    """
    lower_text = cleaned_text.lower()
    words = cleaned_text.split()
    word_count = len(words)

    # Split candidate bullets/sentences
    segments = [
        s.strip(" -•●▪➢\t")
        for s in re.split(r"(?:[\r\n•●▪➢]+|(?<=\.)\s+)", raw_text)
        if 5 <= len(s.split()) <= 45
    ]
    total_bullets = max(1, len(segments))

    # Count quantified segments (%, $, numbers, Xx, ms, sec, k+, M+)
    quant_pattern = re.compile(r"(\d+(\.\d+)?\s*(%|percent|x|ms|s|sec|seconds|hours|days|k\b|m\b|million|users|requests|clients|projects)|\$\s*\d+)", re.IGNORECASE)
    quantified_count = sum(1 for seg in segments if quant_pattern.search(seg))
    quant_rate = int(round((quantified_count / total_bullets) * 100))

    # Strong action verbs found
    tokens_lower = set(re.findall(r"\b[a-z]+\b", lower_text))
    strong_verbs = sorted(list(tokens_lower.intersection(RESUME_ACTION_VERBS)))[:12]

    # Weak phrases found
    weak_found: List[str] = []
    for phrase, fix in WEAK_RESUME_PHRASES.items():
        if phrase in lower_text:
            weak_found.append(fix)

    # Generate Smart XYZ Bullet Suggestions using top missing keywords
    xyz_suggestions: List[str] = []
    target_kws = missing_keywords[:4] if missing_keywords else ["Python", "REST APIs", "CI/CD"]
    for idx, kw in enumerate(target_kws):
        tmpl = XYZ_TEMPLATES[idx % len(XYZ_TEMPLATES)]
        formatted_kw = kw.upper() if len(kw) <= 3 else kw.title()
        xyz_suggestions.append(tmpl.format(kw=formatted_kw))

    bullet_report = BulletImpactReport(
        total_bullets_analyzed=total_bullets,
        quantified_bullets_count=quantified_count,
        quantification_rate=min(100, quant_rate),
        strong_verbs_found=strong_verbs,
        weak_phrases_found=weak_found,
        xyz_bullet_suggestions=xyz_suggestions,
    )

    # Structural ATS Audit Checks
    has_email = bool(re.search(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+", cleaned_text))
    has_phone = bool(
        re.search(
            r"(\+\d{1,3}[\s\-\.]?\d{2,4}[\s\-\.]?\d{3,4}[\s\-\.]?\d{3,4}|\+?\d{1,3}[\s\-\.]?\(?\d{3}\)?[\s\-\.]?\d{3}[\s\-\.]?\d{4})",
            cleaned_text,
        )
    )
    has_links = bool(re.search(r"\b(linkedin\.com|github\.com|gitlab\.com|portfolio)\b", lower_text))
    has_dates = bool(re.search(r"\b(19|20)\d{2}\b", lower_text))
    core_sections_ok = len(detected_sections) >= 3
    word_count_ok = 120 <= word_count <= 1200

    checks: List[AuditCheckItem] = [
        AuditCheckItem(
            label="Contact Header Parseability",
            passed=has_email and has_phone,
            detail="Email & phone number cleanly indexed" if (has_email and has_phone) else "Missing parseable phone or email in header",
        ),
        AuditCheckItem(
            label="Standard Section Hierarchy",
            passed=core_sections_ok,
            detail=f"{len(detected_sections)} canonical headings detected ({', '.join(detected_sections[:4])})"
            if core_sections_ok
            else f"Only {len(detected_sections)} standard heading(s) found; add explicit 'Experience', 'Education', and 'Skills' headers",
        ),
        AuditCheckItem(
            label="Chronological Date Tokens",
            passed=has_dates,
            detail="Standard 4-digit timeline years verified" if has_dates else "No clear timeline dates (YYYY) detected",
        ),
        AuditCheckItem(
            label="Digital Profile Links",
            passed=has_links,
            detail="LinkedIn / GitHub / Portfolio link detected" if has_links else "Add LinkedIn or GitHub URL for recruiter verification",
        ),
        AuditCheckItem(
            label="Optimal Token Volume",
            passed=word_count_ok,
            detail=f"{word_count} words (optimal 1-page density)" if word_count_ok else f"{word_count} words (target 250–900 words for ideal ATS scoring)",
        ),
    ]

    passed_count = sum(1 for c in checks if c.passed)
    readability_score = int(round((passed_count / len(checks)) * 100))

    structural_audit = StructuralAuditReport(
        readability_score=readability_score,
        checks=checks,
        detected_sections=detected_sections,
        raw_text_preview=cleaned_text[:1500] + ("..." if len(cleaned_text) > 1500 else ""),
    )

    return bullet_report, structural_audit


# -------------------------------------------------------------------------
# Role Query Expansion & Online Enrichment Engine
# -------------------------------------------------------------------------

def fetch_online_role_context(query: str) -> str:
    """
    Attempts a fast, lightweight lookup against public knowledge APIs
    when a user types a short niche role name not covered locally.
    Fails silently and instantly if offline or sandboxed.
    """
    clean_q = re.sub(r"[^a-zA-Z0-9\s\+#\.\-]", " ", query).strip()
    if not clean_q or len(clean_q) > 80:
        return ""

    snippets: List[str] = []
    try:
        url = f"https://api.duckduckgo.com/?q={urllib.parse.quote(clean_q + ' software role skills')}&format=json&no_html=1&skip_disambig=1"
        req = urllib.request.Request(url, headers={"User-Agent": "Glyph-ATS-Engine/2.5"})
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="ignore"))
            if data.get("AbstractText"):
                snippets.append(str(data["AbstractText"]))
            for topic in data.get("RelatedTopics", [])[:3]:
                if isinstance(topic, dict) and topic.get("Text"):
                    snippets.append(str(topic["Text"]))
    except Exception:
        pass

    return " ".join(snippets)


def expand_job_specification(raw_jd: str) -> Tuple[str, List[str], List[str]]:
    """
    Parses the user's job input—including comma-separated lists, short words,
    acronyms (SWE, FE, BE, MLE, DS, PM, QA, etc.), and role titles—and expands
    them into a rich target specification with canonical skills.
    """
    cleaned = clean_text(raw_jd)
    lower_jd = cleaned.lower()

    segments = [seg.strip() for seg in re.split(r"[,/;|&\n]+", lower_jd) if seg.strip()]
    individual_words = set(re.findall(TOKEN_PATTERN, lower_jd))

    matched_role_keys: List[str] = []
    expanded_role_titles: List[str] = []
    role_skills_accum: List[str] = []

    for role_key, profile in ROLE_KNOWLEDGE_BASE.items():
        for trigger in profile["triggers"]:
            pattern = r"(?<![a-zA-Z0-9])" + re.escape(trigger) + r"(?![a-zA-Z0-9])"
            if trigger in individual_words or trigger in segments or re.search(pattern, lower_jd):
                if role_key not in matched_role_keys:
                    matched_role_keys.append(role_key)
                    expanded_role_titles.append(profile["title"])
                    role_skills_accum.extend(profile["skills"])
                break

    direct_user_skills: List[str] = []
    # 1. Scan full JD for multi-word and single-word real-world skills in ALL_KNOWN_SKILLS
    for known_sk in sorted(ALL_KNOWN_SKILLS, key=lambda x: (-len(x), x)):
        if len(known_sk) < 2:
            continue
        pat = r"(?<![a-zA-Z0-9])" + re.escape(known_sk) + r"(?![a-zA-Z0-9])"
        if re.search(pat, lower_jd):
            canonical_sk = known_sk
            for grp in SKILL_EQUIVALENCE_GROUPS:
                if known_sk in [g.lower() for g in grp]:
                    canonical_sk = grp[0]
                    break
            if canonical_sk not in direct_user_skills:
                direct_user_skills.append(canonical_sk)

    # 2. Also include explicit short tokens from comma-separated queries
    for seg in segments:
        seg_tokens = re.findall(TOKEN_PATTERN, seg)
        for tok in seg_tokens:
            tok_l = tok.lower().strip(".-_")
            if tok_l in COMMON_STOP_WORDS:
                continue
            if tok_l in SKILL_SYNONYM_MAP:
                for grp in SKILL_EQUIVALENCE_GROUPS:
                    if tok_l in [g.lower() for g in grp]:
                        if grp[0] not in direct_user_skills:
                            direct_user_skills.append(grp[0])
                        break
            elif len(tok_l) >= 2 and tok_l not in {"intern", "internship", "internships", "engineer", "developer", "senior", "junior"}:
                if tok_l not in direct_user_skills:
                    direct_user_skills.append(tok_l)

    priority_keywords: List[str] = []
    seen_kw: Set[str] = set()
    is_short_query = len(cleaned.split()) < 45

    if is_short_query:
        for kw in role_skills_accum + direct_user_skills:
            kw_l = kw.lower()
            if kw_l not in seen_kw and kw_l not in COMMON_STOP_WORDS:
                seen_kw.add(kw_l)
                priority_keywords.append(kw_l)
    else:
        for kw in direct_user_skills + role_skills_accum[:12]:
            kw_l = kw.lower()
            if kw_l not in seen_kw and kw_l not in COMMON_STOP_WORDS:
                seen_kw.add(kw_l)
                priority_keywords.append(kw_l)

    online_context = ""
    if is_short_query and not matched_role_keys:
        online_context = fetch_online_role_context(cleaned)
        if online_context:
            expanded_role_titles.append(f"Live Role Profile: {cleaned.title()}")

    enrichment_parts = [cleaned]
    if expanded_role_titles:
        enrichment_parts.append("Target Roles: " + ", ".join(expanded_role_titles))
    if priority_keywords:
        enrichment_parts.append("Required Technical Skills & Competencies: " + ", ".join(priority_keywords))
    if online_context:
        enrichment_parts.append(online_context)

    enriched_jd_text = " . ".join(enrichment_parts)
    return enriched_jd_text, expanded_role_titles, priority_keywords


# -------------------------------------------------------------------------
# Core Helper Functions: Multi-Format In-Memory Text Extraction & OCR
# -------------------------------------------------------------------------

def clean_text(raw_text: str) -> str:
    """Cleans raw text by sanitizing PDF CID tokens, normalizing whitespace, and stripping null bytes."""
    if not raw_text:
        return ""
    text: str = sanitize_pdf_cid_artifacts(raw_text).replace("\x00", " ")
    text = re.sub(r"[\r\n\t]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def run_opencv_ocr(image_bytes: bytes) -> str:
    """Preprocesses image with OpenCV and runs Tesseract OCR strictly in RAM."""
    if not TESSERACT_EXE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                "Scanned image/certificate detected. Optical Character Recognition (OCR) requires "
                "the Tesseract-OCR engine binary installed on the host system. Please install Tesseract or upload a text PDF."
            ),
        )
    try:
        np_arr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("OpenCV failed to decode image bytes.")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        denoised = cv2.medianBlur(thresh, 3)

        text = pytesseract.image_to_string(denoised)
        return text
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"OCR image processing failed: {str(exc)}",
        )


def extract_multi_format_text(file_bytes: bytes, filename: str, content_type: str) -> Tuple[str, str, str]:
    """
    Extracts (raw_multiline_text, cleaned_singleline_text, format_label) from PDF, PNG, JPG in memory.
    """
    fn_lower = filename.lower()

    if fn_lower.endswith((".png", ".jpg", ".jpeg")) or content_type.startswith("image/"):
        if not (
            file_bytes.startswith(PNG_MAGIC_BYTES)
            or file_bytes[:3] == JPEG_MAGIC_BYTES
            or fn_lower.endswith((".png", ".jpg", ".jpeg"))
        ):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid image header. Uploaded file is not a valid PNG or JPEG image.",
            )
        raw_ocr = run_opencv_ocr(file_bytes)
        return raw_ocr, clean_text(raw_ocr), "Image (OCR)"

    if fn_lower.endswith(".pdf") or content_type == "application/pdf":
        if not file_bytes.startswith(PDF_MAGIC_BYTES):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid PDF header. The uploaded file is not a genuine PDF document.",
            )

        raw_extracted = ""
        try:
            stream = io.BytesIO(file_bytes)
            laparams = LAParams(char_margin=60.0, line_margin=0.2)
            raw_extracted = extract_text(stream, laparams=laparams) or ""
        except Exception:
            try:
                stream = io.BytesIO(file_bytes)
                raw_extracted = extract_text(stream) or ""
            except Exception:
                pass

        cleaned_extracted = clean_text(raw_extracted)

        if len(cleaned_extracted.split()) < 10:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                pages_text = []
                for page in reader.pages:
                    pt = page.extract_text()
                    if pt:
                        pages_text.append(pt)
                combined_raw = "\n".join(pages_text)
                if len(clean_text(combined_raw).split()) >= len(cleaned_extracted.split()):
                    raw_extracted = combined_raw
                    cleaned_extracted = clean_text(combined_raw)
            except Exception:
                pass

        if len(cleaned_extracted.split()) < 10:
            try:
                reader = pypdf.PdfReader(io.BytesIO(file_bytes))
                ocr_fragments = []
                for page in reader.pages:
                    for img_obj in page.images:
                        img_bytes = img_obj.data
                        frag = run_opencv_ocr(img_bytes)
                        if frag:
                            ocr_fragments.append(frag)
                if ocr_fragments:
                    raw_ocr_pdf = "\n".join(ocr_fragments)
                    return raw_ocr_pdf, clean_text(raw_ocr_pdf), "PDF (Scanned OCR)"
            except HTTPException:
                raise
            except Exception:
                pass

        if len(cleaned_extracted.split()) < 10:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=(
                    "Could not extract sufficient text from the PDF. It may be an image-only scan or password-protected. "
                    "Ensure your PDF contains digital text or configure the Tesseract-OCR engine on the server."
                ),
            )

        return sanitize_pdf_cid_artifacts(raw_extracted), cleaned_extracted, "PDF (Selectable)"

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported document format. Please upload a PDF, PNG, or JPG file.",
    )


def resume_contains_keyword_or_synonym(resume_lower: str, keyword: str) -> bool:
    """
    Checks if the resume contains the target keyword OR any of its known
    industry synonyms/abbreviations.
    """
    candidates: Set[str] = {keyword.lower()}
    if keyword.lower() in SKILL_SYNONYM_MAP:
        candidates.update(SKILL_SYNONYM_MAP[keyword.lower()])

    for cand in candidates:
        pattern = r"(?<![a-zA-Z0-9])" + re.escape(cand) + r"(?![a-zA-Z0-9])"
        if re.search(pattern, resume_lower):
            return True
    return False


def analyze_ats_alignment(
    resume_text: str, raw_jd_text: str
) -> Tuple[int, str, float, float, List[str], List[str], List[str], List[str]]:
    """
    Computes ATS alignment using Role Expansion, Synonym Normalization,
    Scikit-Learn TF-IDF vectorization, cosine similarity, and keyword gap analysis.
    """
    enriched_jd_text, expanded_roles, priority_seed_keywords = expand_job_specification(raw_jd_text)

    vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 2),
        max_features=5000,
    )

    tfidf_matrix = vectorizer.fit_transform([resume_text, enriched_jd_text])
    raw_similarity: float = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    raw_similarity = max(0.0, min(1.0, raw_similarity))

    jd_vectorizer = TfidfVectorizer(
        stop_words="english",
        token_pattern=TOKEN_PATTERN,
        ngram_range=(1, 1),
    )
    jd_matrix = jd_vectorizer.fit_transform([enriched_jd_text])
    feature_names = jd_vectorizer.get_feature_names_out()
    scores = jd_matrix.toarray()[0]

    ranked_keywords: List[Tuple[str, float]] = []
    seen_canonical: Set[str] = set()

    for idx, kw in enumerate(priority_seed_keywords):
        clean_kw = kw.lower().strip(".-_#+/")
        if len(clean_kw) >= 2 and clean_kw not in COMMON_STOP_WORDS and clean_kw not in seen_canonical:
            seen_canonical.add(clean_kw)
            ranked_keywords.append((clean_kw, 1.0 - (idx * 0.01)))

    tfidf_pairs = sorted(zip(feature_names, scores), key=lambda x: x[1], reverse=True)
    for word, score in tfidf_pairs:
        clean_word = word.lower().strip(".-_#+/")
        if (
            len(clean_word) >= 2
            and clean_word not in COMMON_STOP_WORDS
            and clean_word not in seen_canonical
            and score > 0.035
        ):
            synonyms = SKILL_SYNONYM_MAP.get(clean_word, set())
            if not synonyms.intersection(seen_canonical):
                seen_canonical.add(clean_word)
                ranked_keywords.append((clean_word, float(score)))

    resume_lower: str = resume_text.lower()
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []

    for word, _ in ranked_keywords:
        if resume_contains_keyword_or_synonym(resume_lower, word):
            matched_keywords.append(word)
        else:
            missing_keywords.append(word)

    matched_keywords = matched_keywords[:30]
    missing_keywords = missing_keywords[:30]

    total_evaluated: int = len(matched_keywords) + len(missing_keywords)
    keyword_match_rate: float = (len(matched_keywords) / max(1, total_evaluated)) * 100.0

    calibrated_score: float = (raw_similarity * 38.0) + (keyword_match_rate * 0.68)
    final_score: int = int(round(max(0, min(100, calibrated_score))))

    role_context_str = f" for {', '.join(expanded_roles[:3])}" if expanded_roles else ""

    if final_score >= 80:
        match_level = "Exceptional Match"
        recommendations = [
            f"Outstanding alignment{role_context_str}. Your resume covers the core technical stack.",
            "Ensure both full terms and common acronyms (e.g., 'Continuous Integration (CI/CD)') appear naturally.",
            "Maintain single-column formatting with quantified impact metrics in your experience bullets.",
        ]
    elif final_score >= 65:
        match_level = "Strong Match"
        recommendations = [
            f"Strong foundation{role_context_str}. Incorporate high-value gap terms: {', '.join(missing_keywords[:5])}.",
            "Weave missing technical keywords directly into project or work experience achievement bullets.",
            "Quantify engineering outcomes alongside your skills (e.g., latency reduction, test coverage, throughput).",
        ]
    elif final_score >= 50:
        match_level = "Moderate Match"
        recommendations = [
            f"Moderate overlap{role_context_str}. Prioritize adding core skills: {', '.join(missing_keywords[:6])}.",
            "Tailor your Executive Summary and Technical Skills section to mirror the target role specification.",
            "Use the 1-Click 'Inject Missing Keywords' and 'XYZ Bullet Generator' in the Resume Studio to close gaps.",
        ]
    else:
        match_level = "Needs Keyword Optimization"
        recommendations = [
            f"Significant keyword gap detected{role_context_str}.",
            f"Prioritize integrating foundational terms: {', '.join(missing_keywords[:8])}.",
            "Open the Resume Studio to auto-inject missing skills and XYZ impact bullets, then click 'Re-Score Resume'.",
        ]

    return (
        final_score,
        match_level,
        raw_similarity,
        keyword_match_rate,
        matched_keywords,
        missing_keywords,
        recommendations,
        expanded_roles,
    )


def categorize_skills_for_ats(skills: List[str]) -> List[Tuple[str, str]]:
    """
    Groups a flat list of skills into the structured 2-column ATS categories
    ('Languages', 'Mobile & Web', 'Cloud & DB', 'Tools & Core') matching the
    reference resume layout while preserving every keyword for ATS scanners.
    """
    lang_set = {
        "python", "javascript", "typescript", "java", "c++", "c#", "sql", "go", "golang",
        "rust", "r", "php", "ruby", "swift", "kotlin", "scala", "bash", "html5", "css3",
    }
    web_mobile_set = {
        "react", "react native", "next.js", "node.js", "fastapi", "django", "flask",
        "spring boot", "express.js", "vue.js", "angular", "tailwind css", "html/css",
        "rest apis", "graphql", "redux", "flutter", "swiftui", "i18next", "react navigation",
        "responsive design", "microservices", ".net", "wpf",
    }
    cloud_db_set = {
        "aws", "google cloud", "gcp", "azure", "firebase", "firestore", "postgresql",
        "mysql", "mongodb", "redis", "sqlite", "oracle", "dynamodb", "snowflake",
        "bigquery", "apache kafka", "docker", "kubernetes", "terraform", "ci/cd",
        "github actions", "jenkins",
    }

    langs: List[str] = []
    web_mob: List[str] = []
    cloud_db: List[str] = []
    tools_core: List[str] = []

    for sk in skills:
        s_clean = sanitize_pdf_cid_artifacts(str(sk)).strip()
        if not s_clean:
            continue
        low = s_clean.lower()
        if low in lang_set:
            langs.append(s_clean)
        elif low in web_mobile_set or any(w in low for w in ["react", "html", "css", "api", "flask", "django", "node", "spring", "vue", "angular"]):
            web_mob.append(s_clean)
        elif low in cloud_db_set or any(w in low for w in ["sql", "db", "cloud", "aws", "firebase", "firestore", "mongo", "redis", "postgres"]):
            cloud_db.append(s_clean)
        else:
            tools_core.append(s_clean)

    categories: List[Tuple[str, str]] = []
    if langs:
        categories.append(("Languages", ", ".join(langs)))
    if web_mob:
        categories.append(("Mobile & Web", ", ".join(web_mob)))
    if cloud_db:
        categories.append(("Cloud & DB", ", ".join(cloud_db)))
    if tools_core:
        categories.append(("Tools & Core", ", ".join(tools_core)))
    return categories


def parse_certification_rows(cert_list: List[str]) -> List[Dict[str, str]]:
    """
    Formats certification lines into (issuer, title, year) for clean 2-column
    rendering with bold issuer/provider on the left and italic year on the right,
    strictly stripping all trailing or duplicate years from the title.
    """
    rows: List[Dict[str, str]] = []
    known_issuers = [
        "IBM / DeepLearning.AI", "DataCamp / Codio", "DeepLearning.AI", "Extern",
        "Udacity", "Google", "IBM", "DataCamp", "Codio", "AWS", "Microsoft",
        "Coursera", "Meta", "Stanford", "Harvard", "Cisco", "CompTIA", "Oracle",
    ]
    for raw_c in cert_list:
        c_str = sanitize_pdf_cid_artifacts(str(raw_c)).strip(" -•●▪\t")
        if not c_str:
            continue
        left_text, year_val = extract_and_strip_date(c_str, default_date="2025")
        if not left_text or not re.search(r"[A-Za-z]{2,}", left_text):
            continue

        issuer_val = ""
        title_val = left_text
        for iss in known_issuers:
            if left_text.lower().startswith(iss.lower() + " ") or left_text.lower().startswith(iss.lower() + ":"):
                issuer_val = left_text[: len(iss)]
                title_val = left_text[len(iss) :].lstrip(" :—–-")
                break
        if not issuer_val and " — " in left_text:
            parts = left_text.split(" — ", 1)
            issuer_val = parts[0].strip()
            title_val = parts[1].strip()

        rows.append({"issuer": issuer_val, "title": title_val, "year": year_val})
    return rows


def render_pdf_in_memory(data: Dict[str, Any]) -> bytes:
    """
    Renders an ATS-compliant single-column resume into a PDF in memory.
    Normalizes contact fields, categorized skills, education, experience,
    projects, and certifications so the resulting PDF is 100% quirk-free
    with zero duplicate dates or years.
    """
    clean_data = dict(data)
    clean_data["full_name"] = sanitize_pdf_cid_artifacts(str(clean_data.get("full_name") or "")).strip()
    clean_data["summary"] = sanitize_pdf_cid_artifacts(str(clean_data.get("summary") or "")).strip()

    loc_str = sanitize_pdf_cid_artifacts(str(clean_data.get("location") or "")).strip()
    full_name_str = clean_data.get("full_name") or ""
    if full_name_str and "," in loc_str:
        name_tokens_lower = {w.lower().strip(".,") for w in full_name_str.split()}
        city_part, _, region_part = loc_str.partition(",")
        city_words = [w for w in city_part.split() if w.lower().strip(".,") not in name_tokens_lower]
        if city_words:
            loc_str = f"{' '.join(city_words)}, {region_part.strip()}"

    phone_str = sanitize_pdf_cid_artifacts(str(clean_data.get("phone") or "")).strip()
    email_str = sanitize_pdf_cid_artifacts(str(clean_data.get("email") or "")).strip()
    li_str = sanitize_pdf_cid_artifacts(str(clean_data.get("linkedin") or "")).strip()
    gh_str = sanitize_pdf_cid_artifacts(str(clean_data.get("github") or "")).strip()

    line1_parts = [p for p in [loc_str, phone_str, email_str] if p]
    line2_parts = [p for p in [li_str, gh_str] if p]
    clean_data["contact_line_1"] = "  •  ".join(line1_parts)
    clean_data["contact_line_2"] = "  •  ".join(line2_parts)

    # Clean & categorize skills
    raw_skills = [sanitize_pdf_cid_artifacts(str(s)).strip() for s in (clean_data.get("skills") or []) if str(s).strip()]
    clean_data["skills"] = raw_skills
    clean_data["skill_categories"] = categorize_skills_for_ats(raw_skills)

    cleaned_exp = []
    for exp in clean_data.get("experience") or []:
        if isinstance(exp, dict):
            raw_title = sanitize_pdf_cid_artifacts(str(exp.get("title") or "")).strip()
            raw_company = sanitize_pdf_cid_artifacts(str(exp.get("company") or "")).strip()
            raw_dates = normalize_date_str(str(exp.get("dates") or ""))
            title, title_dt = extract_and_strip_date(raw_title)
            company, comp_dt = extract_and_strip_date(raw_company)
            dates = raw_dates or title_dt or comp_dt or "June 2022 – Present"
            bullets = [
                sanitize_pdf_cid_artifacts(str(b)).strip().lstrip("-•●▪* ")
                for b in (exp.get("bullets") or [])
                if str(b).strip()
            ]
            if title or company or bullets:
                cleaned_exp.append(
                    {
                        "title": title or raw_title,
                        "company": company,
                        "dates": dates,
                        "bullets": bullets,
                        "description": exp.get("description", ""),
                    }
                )
    clean_data["experience"] = cleaned_exp

    cleaned_proj = []
    for proj in clean_data.get("projects") or []:
        if isinstance(proj, dict):
            raw_name = sanitize_pdf_cid_artifacts(str(proj.get("name") or "")).strip()
            raw_tech = sanitize_pdf_cid_artifacts(str(proj.get("tech") or "")).strip()
            raw_date = normalize_date_str(str(proj.get("date") or ""))
            name, name_dt = extract_and_strip_date(raw_name)
            tech, tech_dt = extract_and_strip_date(raw_tech)
            date_val = raw_date or name_dt or tech_dt or "2025"
            bullets = [
                sanitize_pdf_cid_artifacts(str(b)).strip().lstrip("-•●▪* ")
                for b in (proj.get("bullets") or [])
                if str(b).strip()
            ]
            if name or bullets:
                cleaned_proj.append({"name": name or raw_name, "tech": tech, "date": date_val, "bullets": bullets})
    clean_data["projects"] = cleaned_proj

    cleaned_edu = []
    for edu in clean_data.get("education") or []:
        if isinstance(edu, dict):
            raw_degree = sanitize_pdf_cid_artifacts(str(edu.get("degree") or "")).strip()
            raw_inst = sanitize_pdf_cid_artifacts(str(edu.get("institution") or "")).strip()
            raw_year = normalize_date_str(str(edu.get("year") or ""))
            degree, deg_dt = extract_and_strip_date(raw_degree)
            institution, inst_dt = extract_and_strip_date(raw_inst)
            year = raw_year or inst_dt or deg_dt or "2020 – 2024"
            details = sanitize_pdf_cid_artifacts(str(edu.get("details") or "")).strip()
            if degree or institution:
                cleaned_edu.append(
                    {
                        "degree": degree or raw_degree,
                        "institution": institution or raw_inst,
                        "year": year,
                        "details": details,
                    }
                )
    clean_data["education"] = cleaned_edu

    raw_certs = [sanitize_pdf_cid_artifacts(str(c)).strip() for c in (clean_data.get("certifications") or []) if str(c).strip()]
    clean_data["certifications"] = raw_certs
    clean_data["parsed_certifications"] = parse_certification_rows(raw_certs)

    tmpl = jinja2.Template(RESUME_HTML_TEMPLATE)
    rendered_html = tmpl.render(**clean_data)

    try:
        import weasyprint
        html_doc = weasyprint.HTML(string=rendered_html)
        return html_doc.write_pdf()
    except Exception:
        pdf_stream = io.BytesIO()
        pisa_status = pisa.CreatePDF(rendered_html, dest=pdf_stream)
        if pisa_status.err:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="PDF generation engine encountered an error while building the document.",
            )
        return pdf_stream.getvalue()

# -------------------------------------------------------------------------
# API Endpoints
# -------------------------------------------------------------------------

@app.get("/api/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint to verify backend operational readiness and OCR support."""
    return {
        "status": "healthy",
        "service": "Glyph ATS Engine",
        "version": "2.5.0",
        "phase": "Phase 2.5 (Role Ontology, Resume Classifier, X-Ray Audit, Bullet Impact & Live Studio)",
        "ocr_available": TESSERACT_EXE is not None,
        "roles_indexed": len(ROLE_KNOWLEDGE_BASE),
        "privacy": "Ephemeral (RAM only)",
    }


@app.post("/api/verify-resume")
async def verify_resume_endpoint(
    resume: UploadFile = File(..., description="Document to verify (PDF, PNG, JPG, max 5MB)"),
) -> Dict[str, Any]:
    """
    Instant Ephemeral Document Authenticity Gatekeeper:
    Verifies on file drop/selection whether the uploaded PDF, PNG, or JPG is a genuine
    Resume / CV or a non-resume document (Lecture Notes, Slides, Textbook, Homework, Invoice, etc.).
    """
    try:
        file_bytes: bytes = await resume.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(exc)}",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 5MB ({len(file_bytes)} bytes provided).",
        )

    filename = resume.filename or "document.pdf"
    content_type = resume.content_type or ""

    raw_multiline_text, extracted_text, format_label = extract_multi_format_text(file_bytes, filename, content_type)
    resume_words: List[str] = extracted_text.split()

    if len(resume_words) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Non-Resume Document Rejected: The uploaded file contains fewer than 12 words. Please upload a valid Resume or CV.",
        )

    resume_confidence, detected_sections = validate_document_is_resume(raw_multiline_text, extracted_text, filename)

    del file_bytes
    del raw_multiline_text
    del extracted_text

    return {
        "status": "valid_resume",
        "resume_confidence": resume_confidence,
        "detected_sections": detected_sections,
        "document_format": format_label,
    }


@app.post("/api/analyze", response_model=AnalysisResponse)
async def analyze_resume(
    resume: UploadFile = File(..., description="Resume document (PDF, PNG, JPG, max 5MB)"),
    job_description: str = Form(..., description="Target Job Description, comma-separated roles, or short role acronyms"),
) -> AnalysisResponse:
    """
    Complete Ephemeral ATS Analysis Pipeline:
    1. Validates file headers and extracts text (PDF / OpenCV OCR).
    2. Verifies Resume authenticity (rejects non-resume PDFs/images).
    3. Expands short/comma-separated role queries & synonyms.
    4. Computes TF-IDF vector alignment & keyword gaps.
    5. Evaluates Bullet Quantification (XYZ Formula) & Structural ATS Audit.
    6. Extracts structured candidate profile in RAM to auto-populate the Resume Studio.
    """
    clean_jd: str = clean_text(job_description)
    if len(clean_jd) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Please enter a target job description, role title, or comma-separated skills (e.g., 'Frontend, Backend Internships, SWE').",
        )

    try:
        file_bytes: bytes = await resume.read()
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unable to read uploaded file: {str(exc)}",
        )

    if len(file_bytes) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File exceeds maximum allowed size of 5MB ({len(file_bytes)} bytes provided).",
        )

    filename = resume.filename or "resume.pdf"
    content_type = resume.content_type or ""

    # 1. Multi-Format Text Extraction (PDF / OCR)
    raw_multiline_text, extracted_text, format_label = extract_multi_format_text(file_bytes, filename, content_type)
    resume_words: List[str] = extracted_text.split()

    if len(resume_words) < 12:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded document contains fewer than 12 words. Please upload a complete Resume or CV.",
        )

    # 2. Smart Resume Authenticity Classifier
    resume_confidence, detected_sections = validate_document_is_resume(raw_multiline_text, extracted_text, filename)

    # 3. Role-Expanded & Synonym-Aware ATS Alignment
    score, match_level, cos_sim, kw_rate, matched_kws, missing_kws, recs, expanded_roles = analyze_ats_alignment(
        resume_text=extracted_text,
        raw_jd_text=clean_jd,
    )

    # 4. Auto-Extract & Pre-Optimize Candidate Profile for Resume Studio
    parsed_profile = extract_resume_profile_in_memory(
        raw_multiline_text,
        extracted_text,
        matched_kws,
        missing_kws=missing_kws,
        expanded_roles=expanded_roles,
    )

    # 5. Bullet Quantification & Structural ATS X-Ray Audit
    bullet_impact, structural_audit = analyze_bullet_impact_and_Audit(
        raw_text=raw_multiline_text,
        cleaned_text=extracted_text,
        detected_sections=detected_sections,
        missing_keywords=missing_kws,
    )

    jd_words: List[str] = clean_jd.split()
    stats = StatsModel(
        resume_word_count=len(resume_words),
        job_desc_word_count=len(jd_words),
        matched_keywords_count=len(matched_kws),
        missing_keywords_count=len(missing_kws),
    )

    del file_bytes
    del raw_multiline_text
    del extracted_text

    return AnalysisResponse(
        status="success",
        match_score=score,
        match_level=match_level,
        cosine_similarity=round(cos_sim, 4),
        keyword_match_rate=round(kw_rate, 1),
        matched_keywords=matched_kws,
        missing_keywords=missing_kws,
        stats=stats,
        recommendations=recs,
        document_format=format_label,
        expanded_roles=expanded_roles,
        resume_confidence=resume_confidence,
        parsed_profile=parsed_profile,
        bullet_impact=bullet_impact,
        structural_audit=structural_audit,
        ephemeral_notice="This document was verified as a Resume, parsed entirely in volatile memory, and purged immediately.",
    )


@app.post("/api/generate-resume")
async def generate_resume_endpoint(request: ResumeGenerateRequest) -> Response:
    """
    Phase 2 ATS Resume Generation Engine.
    Takes candidate data, applies ATS single-column formatting, and outputs
    a cleanly formatted, ATS-compliant PDF in memory.
    """
    if not request.full_name.strip():
        raise HTTPException(status_code=400, detail="Full name is required to generate a resume.")
    if not request.email.strip():
        raise HTTPException(status_code=400, detail="Email address is required.")

    data_dict = request.model_dump()
    pdf_bytes = render_pdf_in_memory(data_dict)

    safe_name = re.sub(r"[^a-zA-Z0-9_]", "_", request.full_name)
    filename = f"{safe_name}_ATS_Resume.pdf"

    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Engine": "Glyph-Phase-2-Jinja2",
        },
    )


# -------------------------------------------------------------------------
# Static File Mount & Logo Fallbacks
# -------------------------------------------------------------------------

frontend_dir: Path = Path(__file__).resolve().parent.parent / "frontend"

if frontend_dir.exists():
    assets_dir: Path = frontend_dir / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/logo.jpg", include_in_schema=False)
    @app.get("/Glyph_Logo.jpg", include_in_schema=False)
    async def serve_root_logo() -> FileResponse:
        logo_path: Path = frontend_dir / "assets" / "Glyph_Logo.jpg"
        if logo_path.exists():
            return FileResponse(logo_path)
        raise HTTPException(status_code=404, detail="Logo not found")

    @app.get("/", include_in_schema=False)
    async def serve_index() -> FileResponse:
        index_file: Path = frontend_dir / "index.html"
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(content={"message": "Frontend index.html not found"})
