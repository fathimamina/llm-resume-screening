import os 
import re 
import io 
import json 

import pymupdf 
import streamlit as st 
from docx import Document 
from openai import OpenAI 
from dotenv import load_dotenv 

load_dotenv()

MODEL = "openai/gpt-oss-120b"
BASE_URL = "https://api.groq.com/openai/v1"

#-------------------------Step 1: Extract text ----------------------------------
def extract_resume_text(uploaded_file):
    name = uploaded_file.name.lower()
    data = uploaded_file.getvalue()

    if name.endswith('.pdf'):
        doc = pymupdf.open(stream=data, filetype="pdf")
        text = "\n".join(page.get_text() for page in doc)
        doc.close()
    elif name.endswith(".docx"):
        doc = Document(io.BytesIO(data))
        text = "\n".join(p.text for p in doc.paragraphs)
    else:
        raise ValueError("Unsupported file format")
    return text  

#----------------------- Step 2: Preprocessing ---------------------------------------
def clean_text(text):
    text = re.sub(r'\s+',' ', text)
    return text.strip()

#----------------------- Step 3: LLM -------------------------------------------------
def get_api_key():
    key = os.getenv('GROQ_API_KEY')
    if not key:
        try:
            key = st.secrets['GROQ_API_KEY'] 
        except Exception:
            key = None 
    return key 


def ask_llm_json(client, prompt):
    response = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": prompt}],
    temperature=0,                      # was: temparature
    response_format={"type": "json_object"}
)
    return json.loads(response.choices[0].message.content)

def get_resume_data(client, cleaned_resume):
    prompt = f"""
Analyze the following resume.

Extract:
1. name
2. technical_skills (short names only, like "Python", "SQL", "PyTorch")
3. soft_skills
4. education
5. experience
6. projects
7. certifications

Return ONLY valid JSON.

Resume:
{cleaned_resume}
""" 
    return ask_llm_json(client, prompt)

def get_job_data(client, job_description):
    prompt = f"""
Analyze the following job description.

Extract:
1. job_title
2. technical_skills (short names only, like "Python", "SQL", "PyTorch")
3. soft_skills
4. education
5. experience
6. responsibilities

Return ONLY valid JSON.

Job Description:
{job_description}
"""
    return ask_llm_json(client, prompt) 

def get_suggestions(client, job_data, matched_skills, missing_skills, ats_score):
    prompt = f"""
A candidate applied for the job: {job_data.get("job_title", "the job")}.

Matched skills: {matched_skills}
Missing skills: {missing_skills}
ATS score: {ats_score}%

Give 3 short, practical tips to improve this resume for the job.
"""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    return response.choices[0].message.content

# ---------- Step 4: ATS scoring ----------
def calculate_ats(resume_data, job_data, cleaned_resume):
    resume_skills = [str(s).lower() for s in resume_data.get("technical_skills", [])]
    job_skills = [str(s).lower() for s in job_data.get("technical_skills", [])]
    resume_text_lower = cleaned_resume.lower()

    matched_skills = []
    missing_skills = []

    for skill in job_skills:
        if skill in resume_skills or skill in resume_text_lower:
            matched_skills.append(skill)
        else:
            missing_skills.append(skill)

    if len(job_skills) == 0:
        ats_score = 0.0
    else:
        ats_score = round(len(matched_skills) / len(job_skills) * 100, 1)

    return matched_skills, missing_skills, ats_score



# ---------- Streamlit UI ----------
def main():
    st.set_page_config(page_title="Resume Screening", page_icon="📄", layout="wide")
    st.title("📄 Resume Screening (ATS Score)")
    st.caption("Upload a resume, paste a job description, and see how well they match.")

    # Sidebar: API key
    api_key = get_api_key()
    if not api_key:
        api_key = st.sidebar.text_input("Groq API Key", type="password")
    st.sidebar.write("Model:", MODEL)

    # Inputs
    col1, col2 = st.columns(2)
    with col1:
        uploaded_file = st.file_uploader("Upload resume (PDF or DOCX)", type=["pdf", "docx"])
    with col2:
        job_description = st.text_area("Paste job description", height=250)

    if not st.button("Analyze", type="primary"):
        return

    # Input checks
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
        return
    if uploaded_file is None:
        st.error("Please upload a resume.")
        return
    if not job_description.strip():
        st.error("Please paste a job description.")
        return

    try:
        client = OpenAI(api_key=api_key, base_url=BASE_URL)

        with st.spinner("Reading resume..."):
            resume_text = extract_resume_text(uploaded_file)
            cleaned_resume = clean_text(resume_text)

        if len(cleaned_resume) < 50:
            st.error("Could not read text from this file. It may be a scanned image PDF.")
            return

        with st.spinner("Analyzing resume and job description..."):
            resume_data = get_resume_data(client, cleaned_resume)
            job_data = get_job_data(client, job_description)

        matched_skills, missing_skills, ats_score = calculate_ats(resume_data, job_data, cleaned_resume)

        with st.spinner("Writing suggestions..."):
            suggestions = get_suggestions(client, job_data, matched_skills, missing_skills, ats_score)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
        return

    # Results
    st.divider()
    st.subheader("Result")
    st.metric("ATS Score", f"{ats_score}%")
    st.progress(min(int(ats_score), 100))

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Matched skills**")
        st.success(", ".join(matched_skills) if matched_skills else "None")
    with c2:
        st.markdown("**Missing skills**")
        st.error(", ".join(missing_skills) if missing_skills else "None")

    st.subheader("Suggestions")
    st.write(suggestions)

    with st.expander("Extracted resume data"):
        st.json(resume_data)
    with st.expander("Extracted job data"):
        st.json(job_data)


if __name__ == "__main__":
    main()