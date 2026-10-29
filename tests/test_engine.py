from contextlib import contextmanager
from fastapi import HTTPException
from fastapi.testclient import TestClient

from backend.main import (
    app,
    validate_document_is_resume,
    expand_job_specification,
    resume_contains_keyword_or_synonym,
    analyze_ats_alignment,
    extract_resume_profile_in_memory,
    extract_and_strip_date,
    render_pdf_in_memory,
    sanitize_pdf_cid_artifacts,
    categorize_skills_for_ats,
)

client = TestClient(app)


@contextmanager
def raises(exc_type):
    """Zero-dependency context manager for exception assertion (compatible with pytest.raises)."""
    class ExcHolder:
        def __init__(self):
            self.value = None

    holder = ExcHolder()
    try:
        yield holder
    except exc_type as exc:
        holder.value = exc
    else:
        raise AssertionError(f"Expected exception {exc_type.__name__} was not raised")


SAMPLE_VALID_RESUME = """
Alex Vance
San Francisco, CA | +1 (555) 019-2834 | alex.vance@glyph.dev | linkedin.com/in/alexvance | github.com/alexvance

Professional Summary
Senior Software Engineer with 5+ years of experience building distributed backend services, REST APIs, and NLP pipelines.

Technical Skills
Python, FastAPI, JS, TS, React, Docker, K8s, PostgreSQL, Redis, CI/CD, Git, Linux, OOP, DSA

Professional Experience
Senior Backend Engineer — Starlight Systems (2022 - Present)
• Architected resilient microservices with FastAPI handling 10M+ requests daily, reducing latency by 35%.
• Automated CI/CD deployment pipelines on Kubernetes, cutting release cycles by 40%.

Education
B.S. in Computer Science — University of Washington (2017 - 2021)
"""


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["version"] == "2.5.0"


def test_resume_classifier_accepts_valid_resume():
    confidence, sections = validate_document_is_resume(
        SAMPLE_VALID_RESUME, SAMPLE_VALID_RESUME, "Alex_Vance_Resume.pdf"
    )
    assert confidence >= 75
    assert "Education" in sections
    assert "Experience" in sections


def test_resume_classifier_rejects_lecture_notes():
    lecture_notes_text = """
    University of Washington - Department of Computer Science
    Lecture 4: Distributed Systems, Python Microservices & REST APIs
    Learning Objectives:
    In this lecture we will explore how distributed systems and microservices communicate.
    Suppose that a client sends a request to a backend server. For example, we can design
    a fault-tolerant architecture using Python and Docker.
    Key Concepts & Definitions:
    1. Latency vs Throughput in network protocols.
    2. Consensus algorithms and replication state machines.
    Lecture Summary:
    Students will develop practical skills for the course project and upcoming midterm review.
    """
    with raises(HTTPException) as exc_info:
        validate_document_is_resume(lecture_notes_text, lecture_notes_text, "document_scan.pdf")
    assert exc_info.value.status_code == 400
    assert "Non-Resume Document Rejected" in exc_info.value.detail


def test_resume_classifier_rejects_invoice():
    fake_invoice = """
    INVOICE #98142
    Bill To: Acme Corp
    Ship To: 123 Industrial Way
    Item Description: Cloud Hosting Subscription
    Subtotal: $1,200.00
    Tax Invoice: $96.00
    Total Due: $1,296.00
    Payment Method: Wire Transfer / Balance Due within 30 days.
    """
    with raises(HTTPException) as exc_info:
        validate_document_is_resume(fake_invoice, fake_invoice, "invoice_october.pdf")
    assert exc_info.value.status_code == 400
    assert "Non-Resume Document Rejected" in exc_info.value.detail


def test_resume_classifier_rejects_homework_exam():
    fake_homework = """
    CS 101 Midterm Exam — Fall Semester
    Course Syllabus & Grading Policy
    Office Hours: Tue/Thu 2pm
    Total Marks: 100
    Answer all questions in the space provided.
    Problem Set 1:
    Question 1: Prove that the time complexity of merge sort is O(n log n).
    """
    with raises(HTTPException) as exc_info:
        validate_document_is_resume(fake_homework, fake_homework, "midterm_exam.pdf")
    assert exc_info.value.status_code == 400
    assert "Non-Resume Document Rejected" in exc_info.value.detail


def test_short_comma_separated_role_expansion():
    raw_query = "Frontend, Backend Internships , SWE"
    enriched_text, expanded_roles, priority_kws = expand_job_specification(raw_query)

    assert "Software Engineer (SWE)" in expanded_roles
    assert "Frontend / Web Engineer" in expanded_roles
    assert "Backend / API Engineer" in expanded_roles
    assert "Software / Tech Internship & Early Career" in expanded_roles
    assert "javascript" in priority_kws
    assert "python" in priority_kws
    assert "data structures" in priority_kws


def test_synonym_and_acronym_equivalence():
    resume_lower = SAMPLE_VALID_RESUME.lower()
    # Resume has 'js', 'ts', 'k8s', 'oop', 'dsa' -> should match full canonical terms
    assert resume_contains_keyword_or_synonym(resume_lower, "javascript") is True
    assert resume_contains_keyword_or_synonym(resume_lower, "typescript") is True
    assert resume_contains_keyword_or_synonym(resume_lower, "kubernetes") is True
    assert resume_contains_keyword_or_synonym(resume_lower, "object-oriented programming") is True
    assert resume_contains_keyword_or_synonym(resume_lower, "data structures") is True


def test_auto_profile_extraction():
    profile = extract_resume_profile_in_memory(SAMPLE_VALID_RESUME, SAMPLE_VALID_RESUME, ["python", "fastapi"])
    assert profile.email == "alex.vance@glyph.dev"
    assert "linkedin.com/in/alexvance" in profile.linkedin
    assert len(profile.skills) > 0
    assert len(profile.experience) > 0


def test_in_memory_pdf_compiler():
    payload = {
        "full_name": "Alex Vance",
        "email": "alex.vance@glyph.dev",
        "phone": "+1 (555) 019-2834",
        "location": "San Francisco, CA",
        "linkedin": "linkedin.com/in/alexvance",
        "github": "github.com/alexvance",
        "summary": "Senior Software Engineer specializing in Python and distributed systems.",
        "skills": ["Python", "FastAPI", "Docker", "Kubernetes"],
        "experience": [
            {
                "title": "Senior Backend Engineer",
                "company": "Starlight Systems",
                "dates": "2022 - Present",
                "bullets": ["Architected resilient microservices handling 10M requests daily."],
            }
        ],
        "education": [
            {
                "degree": "B.S. in Computer Science",
                "institution": "University of Washington",
                "year": "2021",
            }
        ],
    }
    pdf_bytes = render_pdf_in_memory(payload)
    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 500


def test_cid_artifact_sanitization_and_skill_categorization():
    dirty_text = "(cid:136) Analyzed global air quality; (cid:22) sprint with gami(cid:12)cation and o(cid:11)ine persistence."
    cleaned = sanitize_pdf_cid_artifacts(dirty_text)
    assert "(cid:" not in cleaned
    assert "gamification" in cleaned
    assert "offline" in cleaned

    cats = categorize_skills_for_ats(["Python", "JavaScript", "React", "Firebase", "Git"])
    cat_names = [c[0] for c in cats]
    assert "Languages" in cat_names
    assert "Mobile & Web" in cat_names
    assert "Cloud & DB" in cat_names
    assert "Tools & Core" in cat_names


def test_date_deduplication_and_multi_entry_resume_parsing():
    # 1. Verify space-separated LaTeX year ranges and duplicate dates are stripped cleanly
    txt1, dt1 = extract_and_strip_date("University of Karachi 2024   2028")
    assert txt1 == "University of Karachi"
    assert dt1 == "2024 – 2028"

    txt2, dt2 = extract_and_strip_date("Karachi Complaint Portal — Civic-Tech Mobile App 2025 (2025)")
    assert txt2 == "Karachi Complaint Portal — Civic-Tech Mobile App"
    assert dt2 == "2025"

    txt3, dt3 = extract_and_strip_date("C# • .NET 10 • WPF")
    assert ".NET 10" in txt3
    assert dt3 == ""

    # 2. Verify full multi-entry reference resume parsing (2 Educations, 3 Projects, wrapped bullets, 5 Certifications)
    ref_resume = """Syed Haziq Zia Naqvi
Karachi, Pakistan ˆ +92 300 123 4567 ˆ syedhaziq@example.com
linkedin.com/in/syed-haziq-zia-naqvi ˆ github.com/syedhaziqzia
SUMMARY
CS undergraduate who has shipped a civic-tech Android app (React Native + Firebase) with real-time maps, gami cation,
and trilingual UI.
EDUCATION
University of Karachi 2024   2028
Bachelor of Science in Computer Science
Govt. Degree College Majeed SRE 2021   2023
Higher Secondary Certi cate (Pre-Engineering)
EXPERIENCE
Data Storytelling Sprint Extern Aug 2026
Extern, Inc.   sprint conducted in collaboration with PwC volunteers (The Human Skills Project) & Canva
ˆ Analyzed global air quality data (23,000+ cities) to identify PM2.5 as the dominant AQI driver; surfaced Pakistan as
the world's highest-polluted country and presented  ndings via Canva dashboard to PwC volunteers.
PROJECTS
Karachi Complaint Portal   Civic-Tech Mobile App 2025
React Native ˆ Firebase ˆ Firestore ˆ JavaScript
ˆ Designed and shipped a full-featured Android app enabling citizens to report civic issues (road damage, sewage,
encroachments) with GPS-pinned photo evidence on a real-time Firestore-backed map.
MindEase   Mental Health Platform 2025
Python ˆ Flask ˆ MySQL ˆ HTML/CSS
ˆ Role-based web app (Admin, Counselor, User); 15-table normalized MySQL schema with full CRUD, JOINs, CAS\ufffdCADE deletion, parameterized queries, and PBKDF2+SHA256 password hashing.
FuelQ   Queuing Simulator 2025
C# ˆ .NET 10 ˆ WPF
ˆ Lead developer: discrete-event simulator for Shell Maskan petrol pump; implemented 8 Kendall queuing models
validated against a 320-vehicle empirical dataset using Chi-Square testing.
TECHNICAL SKILLS
Languages JavaScript, Python, Java, C#, SQL
Mobile & Web React Native, Flask, React Navigation, i18next, HTML/CSS
Cloud & DB Firebase (Firestore, Auth, Storage), MySQL
Tools Linux (CLI), .NET / WPF, Android Studio, VirtualBox, Git
CERTIFICATIONS
Extern Data Storytelling & Visualization Sprint 2026
Udacity AWS AI Practitioner Challenge 2026
"""
    prof = extract_resume_profile_in_memory(ref_resume, ref_resume, ["python", "javascript", "sql"])
    assert prof.full_name == "Syed Haziq Zia Naqvi"
    assert prof.location == "Karachi, Pakistan"
    assert "Zia Naqvi" not in prof.location

    assert len(prof.education) == 2
    assert prof.education[0].institution == "University of Karachi"
    assert prof.education[0].year == "2024 – 2028"
    assert "2024" not in prof.education[0].institution
    assert prof.education[1].institution == "Govt. Degree College Majeed SRE"
    assert prof.education[1].year == "2021 – 2023"

    assert len(prof.projects) == 3
    assert "2025" not in prof.projects[0].name
    assert prof.projects[1].name == "MindEase — Mental Health Platform"
    assert "CASCADE" in prof.projects[1].bullets[0]
    assert prof.projects[2].name == "FuelQ — Queuing Simulator"

    assert "surfaced Pakistan as the world's highest-polluted country" in prof.experience[0].bullets[0]


if __name__ == "__main__":
    import inspect
    current_module = inspect.getmodule(inspect.currentframe())
    test_functions = [
        obj for name, obj in inspect.getmembers(current_module)
        if name.startswith("test_") and inspect.isfunction(obj)
    ]
    passed = 0
    print(f"\n--- Running {len(test_functions)} Glyph ATS Engine Tests ---")
    for test_fn in test_functions:
        try:
            test_fn()
            print(f"  [PASS] {test_fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"  [FAIL] {test_fn.__name__}: {e}")
    print(f"\nResult: {passed}/{len(test_functions)} tests passed.\n")



