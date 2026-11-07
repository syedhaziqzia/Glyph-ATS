# Glyph ⚡ (v2.5)
> **Private, Fast ATS Resume Analyzer, Parser Audit & Live Resume Studio**

![Glyph Preview](frontend/assets/readme.png)

Glyph bridges the gap between candidate resumes and modern Applicant Tracking Systems (such as Workday, Greenhouse, and Lever). It checks PDF and image resumes entirely in **system memory (RAM)**—combining **Smart Role Matching**, **Synonym and Acronym Detection**, **Keyword Coverage Scoring**, **Quantified Achievement Bullet Suggestions**, and a **Live Split-Screen Resume Builder** with **zero data saved to disk**.

---

## 💡 Why I Built Glyph (My Personal Experience)

I built **Glyph** out of my own frustration while applying for internships and software roles and testing my resume online:

1. **Every "Good" Tool Was Locked Behind Paywalls:** Almost every online ATS checker locked useful feedback—like missing keywords, formatting flags, and resume re-scoring—behind `$29–$49/month` subscriptions at the exact time when students and job seekers shouldn't be forced to pay just to apply for jobs.
2. **"Free" Tools Were Crippled or Had Strict Limits:** Tools claiming to be free either blurred out missing skills, removed key features, or put strict timers that allowed scanning only once unless you upgraded or made multiple accounts.
3. **Zero Privacy & Sketchy Data Harvesting:** A resume holds your personal information—your full name, phone number, email address, location, education, and work history. Many online scanners quietly upload, store, and harvest resumes to train models or sell candidate data to third parties.

I wanted a tool that anyone applying for jobs could genuinely trust and use without restrictions. **Glyph** is **100% free**, has **no scan limits or timers**, **requires no account**, expands short role titles using industry standards, and processes every document (`PDF`, `PNG`, `JPG`) **only in memory** so your personal data is never saved or shared.

---

## 🌟 What Glyph Does (v2.5)

- **🛡️ Real Resume vs. Non-Resume Checker (PDF, PNG, JPG):** Verifies that uploaded files are genuine resumes or CVs (checking for contact details, section titles, and typical resume structure) while automatically rejecting lecture notes, textbooks, homework, invoices, and research papers.
- **🧠 Smart Role Expansion:** Accepts full job descriptions or short role titles (such as `Frontend, Backend Internships, SWE, ML`). It automatically expands them into key skill benchmarks across software, data, product, design, and cybersecurity roles.
- **🔄 Acronym and Synonym Recognition:** Understands short and full technical terms interchangeably (e.g., `JS` ↔ `JavaScript`, `TS` ↔ `TypeScript`, `K8s` ↔ `Kubernetes`, `Postgres` ↔ `PostgreSQL`, `OOP` ↔ `Object-Oriented Programming`, `AWS` ↔ `Amazon Web Services`).
- **🔬 Parser Audit & X-Ray View:** Audits contact headers, section formatting, dates, and profile links. Includes an **X-Ray View** that reveals the exact raw text stream extracted from your document.
- **📈 Quantified Bullet Impact & XYZ Formula:** Checks what percentage of your resume bullets include concrete numbers and metrics (`%`, `$`, scale, latency), flags passive phrasing, and suggests tailored achievement bullets for missing skills.
- **📄 Split-Screen Resume Studio & Live Re-Scoring:** Automatically populates your uploaded resume into a clean split-screen editor with a live paper preview, 1-click missing skill injection, and a **"Re-Score in RAM"** button to verify your score improvement before exporting an ATS-ready PDF.

---

## 🏗️ Project Structure

```text
Glyph/
├── backend/
│   ├── main.py              # FastAPI ATS engine, parser audit & in-memory PDF builder
│   └── requirements.txt     # Python backend dependencies
├── frontend/
│   ├── assets/
│   │   ├── readme.png       # UI showcase preview
│   │   ├── glyph_emblem.png # Isolated logo emblem
│   │   └── Glyph_Logo.jpg   # Brand artwork
│   └── index.html           # Deep navy interface, match report & live studio
├── tests/
│   └── test_engine.py       # Automated test suite (ontology, synonyms, PDF compiler)
├── LICENSE                  # MIT License
└── README.md                # Project documentation & local setup instructions
```

---

## 🚀 Quick Start (Run Locally)

1. **Install Dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```
2. **Start the Server:**
   ```bash
   python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
   ```
3. **Open Glyph:**
   Visit **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser.

---

## 🧪 Running Automated Tests

Run the test suite to verify the parser, ontology, and PDF generator:

```bash
python tests/test_engine.py
```
*(or via `pytest tests/test_engine.py -v`)*

---

## 📄 License
MIT License © 2026 Syed Haziq Zia Naqvi. Engineered with privacy and care.
