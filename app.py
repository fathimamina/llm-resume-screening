"""Resume Screener: Streamlit app that scores resumes against a job description with an LLM."""
import io
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor

import anthropic
import pandas as pd
import streamlit as st
from docx import Document
from dotenv import load_dotenv
from pypdf import PdfReader

load_dotenv()

MODEL = os.getenv("ANTHROPIC_MODEL", "claude-sonnet-5-5")
MAX_CHARS = 15000
MAX_BYTES = 5 * 1024 * 1024
WORKERS = 4

SYSTEM_PROMPT = """You are an experienced recruiter screening a resume against a job description.
Judge ONLY on skills, experience, achievements and education relevant to the role.
Ignore name, gender, age, nationality, photo, marital status and any other protected characteristic.
Do not invent facts: only use what the resume says.

Respond with a single JSON object and nothing else, using exactly these keys:
{
  "candidate_name": string (as written on the resume, or "Unknown"),
  "score": integer 0-100 (overall fit for this role),
  "recommendation": "strong_match" | "possible_match" | "weak_match",
  "matched_skills": [string],
  "missing_skills": [string],
  "years_experience": string (relevant experience, e.g. "5 years" or "unclear"),
  "summary": string (2-3 sentences explaining the score)
}"""

LABELS = {"strong_match": "Strong match", "possible_match": "Possible match", "weak_match": "Weak match"}


def extract_text(filename: str, data: bytes) -> str:
    name = filename.lower()
    if name.endswith(".pdf"):
        reader = PdfReader(io.BytesIO(data))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if name.endswith(".docx"):
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if name.endswith(".txt"):
        return data.decode("utf-8", errors="ignore")
    raise ValueError("Unsupported file type. Use PDF, DOCX or TXT.")


def parse_json(text: str) -> dict:
    text = re.sub(r"```(?:json)?", "", text).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("Model did not return JSON")
    return json.loads(text[start : end + 1])


def score_resume(client, job_description: str, filename: str, data: bytes) -> dict:
    try:
        if len(data) > MAX_BYTES:
            raise ValueError("File is larger than 5 MB.")
        text = extract_text(filename, data).strip()
        if len(text) < 50:
            raise ValueError("Could not read text from this file (scanned image?).")
        resp = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": f"<job_description>\n{job_description}\n</job_description>\n\n"
                           f"<resume>\n{text[:MAX_CHARS]}\n</resume>",
            }],
        )
        result = parse_json("".join(b.text for b in resp.content if b.type == "text"))
        result["score"] = max(0, min(100, int(result.get("score", 0))))
        result["filename"] = filename
        return result
    except Exception as e:  # one bad file should not fail the batch
        return {"filename": filename, "error": str(e), "score": -1}


# ---------- UI ----------
st.set_page_config(page_title="Resume Screener", layout="wide")
st.title("Resume Screener")
st.caption("Paste a job description, upload resumes, and get candidates ranked by fit.")

api_key = os.getenv("ANTHROPIC_API_KEY", "")
if not api_key or api_key == "your-key-here":
    api_key = st.sidebar.text_input("Anthropic API key", type="password")

left, right = st.columns([1, 2], gap="large")

with left:
    jd = st.text_area("Job description", height=260,
                      placeholder="Role, required skills, years of experience...")
    files = st.file_uploader("Resumes (PDF, DOCX or TXT)", type=["pdf", "docx", "txt"],
                             accept_multiple_files=True)
    run = st.button("Screen resumes", type="primary", use_container_width=True)

with right:
    if run:
        if not api_key:
            st.error("Add your Anthropic API key in .env or the sidebar.")
        elif len(jd.strip()) < 30:
            st.error("The job description is too short. Add the role and required skills.")
        elif not files:
            st.error("Upload at least one resume.")
        else:
            client = anthropic.Anthropic(api_key=api_key)
            payload = [(f.name, f.getvalue()) for f in files]
            bar = st.progress(0.0, text="Screening...")
            results = []
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                futures = [pool.submit(score_resume, client, jd, n, d) for n, d in payload]
                for i, fut in enumerate(futures, start=1):
                    results.append(fut.result())
                    bar.progress(i / len(futures), text=f"Screened {i} of {len(futures)}")
            bar.empty()
            results.sort(key=lambda r: r["score"], reverse=True)
            st.session_state["results"] = results

    results = st.session_state.get("results")
    if not results:
        st.info("Ranked candidates will appear here.")
    else:
        ok = [r for r in results if "error" not in r]
        if ok:
            df = pd.DataFrame([{
                "Candidate": r["candidate_name"], "Score": r["score"],
                "Recommendation": LABELS.get(r["recommendation"], r["recommendation"]),
                "Experience": r["years_experience"],
                "Matched skills": ", ".join(r["matched_skills"]),
                "Missing skills": ", ".join(r["missing_skills"]),
                "File": r["filename"],
            } for r in ok])
            st.dataframe(df, hide_index=True, use_container_width=True,
                         column_config={"Score": st.column_config.ProgressColumn(
                             "Score", min_value=0, max_value=100, format="%d")})
            st.download_button("Download results (CSV)", df.to_csv(index=False).encode("utf-8"),
                               "screening_results.csv", "text/csv")

        for i, r in enumerate(results):
            if "error" in r:
                with st.expander(f"{r['filename']}: could not screen"):
                    st.error(r["error"])
            else:
                title = f"{r['score']}  |  {r['candidate_name']}  |  {LABELS.get(r['recommendation'], '')}"
                with st.expander(title, expanded=(i == 0)):
                    st.write(r["summary"])
                    st.caption(f"Relevant experience: {r['years_experience']}  |  File: {r['filename']}")
                    st.markdown("**Matched skills:** " + (", ".join(r["matched_skills"]) or "None listed"))
                    st.markdown("**Missing skills:** " + (", ".join(r["missing_skills"]) or "None listed"))