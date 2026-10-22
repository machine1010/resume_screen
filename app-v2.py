import datetime
import base64
from typing import List

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# 1. Output schema (matches the prompt below)
# ---------------------------------------------------------------------------
class AuditCheck(BaseModel):
    id: int = Field(description="Check number, 1 to 11.")
    name: str = Field(description="Check name, e.g. 'Career Gaps'.")
    status: str = Field(description="GREEN, AMBER or RED.")
    finding: str = Field(description="Short finding with evidence from the resume.")


class CandidateAudit(BaseModel):
    full_name: str
    email: str
    phone: str
    current_title: str
    current_employer: str
    notice_period: str = Field(description="'Immediate' if no current employer on the resume, else as stated, else NOT_STATED.")
    total_experience_years: float = Field(description="Number only.")
    relevant_ai_experience_years: float = Field(description="Number only.")
    top_skills: List[str]
    missing_must_haves: List[str]
    achievements: List[str] = Field(description="Top 3, keep numbers exactly as written.")
    discrepancies: List[str]
    checks: List[AuditCheck] = Field(description="Exactly 11 entries, one per check, in order.")
    interview_questions: List[str] = Field(description="3 questions targeting the weakest claims.")
    recommendation: str = Field(description="PASS, PROCEED_WITH_CAUTION, FLAG_FOR_AUDIT or REJECT.")
    reason: str


# ---------------------------------------------------------------------------
# 2. System instruction
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert HR Forensic Auditor and Resume Authenticity Specialist.
Analyze the candidate's resume PDF and execute live web lookups where necessary. Extract the candidate's full name.

AUDIT CHECKS TO EXECUTE:
1. Contact Info Verification (Email and Phone validation).
3. Career Gap Analysis (Identify gaps > 6 months for most recent Career Gap).
4. Education & Work Experience Overlap (Degrees overlapping full-time jobs).
5. Company-to-Company Tenure Overlap (Concurrent full-time employment).
6. Career Trajectory & Title Escalation (Unrealistic title jumps).
7. Recent Skills Application Check (Last 1 Year): Verify skills(snowflake and DBT(data build tool) ) appear in projects worked on in the last 1 year.
8. Experience Range Filter (5.7 to 10 Years): Total experience MUST be between 5.7 and 10 years. Flag as RED_FLAG if outside 5.7-10 years.
9. Blacklisted Project Audit: Compare listed projects against this blacklist:
    {blacklisted_projects_str}
    Flag as RED_FLAG if any match is found.
10. Overall Candidate Risk Score & Recommendation (PASS / PROCEED_WITH_CAUTION / FLAG_FOR_AUDIT / REJECT).
"""

MODEL_NAME = "gemini-3.8-flash"


# ---------------------------------------------------------------------------
# 3. Processing
# ---------------------------------------------------------------------------
def process_candidate_resumes(api_key: str, use_vertex: bool, uploaded_files_list: List,
                              blacklisted_projects: List[str]):
    # Inline PDF bytes work with both key types and avoid the Files API upload call.
    client = genai.Client(vertexai=use_vertex, api_key=api_key)

    blacklist_formatted = "\n".join(f"    - {proj}" for proj in blacklisted_projects) or "    - (none)"
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        current_date=datetime.date.today().strftime("%d %B %Y"),
        blacklisted_projects_str=blacklist_formatted,
    )

    summary_rows, check_rows = [], []

    for uploaded_file in uploaded_files_list:
        file_name = uploaded_file.name
        try:
            pdf_b64 = base64.b64encode(uploaded_file.getvalue()).decode("utf-8")
            pdf_part = {
                "inline_data": {
                    "mime_type": "application/pdf",
                    "data": pdf_b64
                }
            }

            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=[pdf_part, f"Run all 11 checks on the resume '{file_name}'."],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=CandidateAudit,
                    temperature=0,
                ),
            )
            a = CandidateAudit.model_validate_json(response.text)
            statuses = {c.id: c.status for c in a.checks}

            summary_rows.append({
                "File Name": file_name,
                "Name": a.full_name,
                "Email": a.email,
                "Phone": a.phone,
                "Current Title": a.current_title,
                "Current Employer": a.current_employer,
                "Notice Period": a.notice_period,
                "Total Experience": a.total_experience_years,
                "Relevant AI Experience": a.relevant_ai_experience_years,
                "Recommendation": a.recommendation,
                "Reason": a.reason,
                "RED Checks": sum(s == "RED" for s in statuses.values()),
                "AMBER Checks": sum(s == "AMBER" for s in statuses.values()),
                "Top Skills": ", ".join(a.top_skills),
                "Missing Must-Haves": ", ".join(a.missing_must_haves),
                "Achievements": " | ".join(a.achievements),
                "Discrepancies": " | ".join(a.discrepancies),
                "Interview Questions": " | ".join(a.interview_questions),
            })
            for c in a.checks:
                check_rows.append({
                    "File Name": file_name, "Name": a.full_name,
                    "Check #": c.id, "Check": c.name, "Status": c.status, "Finding": c.finding,
                })

        except Exception as e:
            summary_rows.append({"File Name": file_name, "Name": "Unknown",
                                 "Recommendation": "ERROR", "Reason": f"Failed to audit: {e}"})

    return pd.DataFrame(summary_rows), pd.DataFrame(check_rows)


# ---------------------------------------------------------------------------
# 4. Streamlit UI
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume Auditor", layout="wide")
st.title("🕵️‍♂️ HR Resume Scanner — AI Engineer")
st.markdown("Upload candidate resumes (PDFs) to run the 11-point audit and download the results as CSV.")

with st.sidebar:
    st.header("Configuration")
    key_type = st.radio("Key type", ["Vertex AI (Express) key", "Google AI Studio key"])
    api_key_input = st.text_input("API Key", type="password")

    st.subheader("Blacklisted Projects")
    default_blacklist = (
        "Project Titan - Online Banking System (Proxy Template)\n"
        "ABC Technologies E-Commerce Management System\n"
        "Global Logistics Tracking Hub (Training Scam)\n"
        "Healthcare Patient Portal v2.0 (Common Copy-Paste Project)"
    )
    blacklist_text = st.text_area("Enter one project per line:", value=default_blacklist, height=200)

uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Run Audit"):
    if not api_key_input:
        st.error("Please provide an API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        blacklist_projects = [line.strip() for line in blacklist_text.split("\n") if line.strip()]
        with st.spinner(f"Auditing {len(uploaded_files)} resume(s)..."):
            summary_df, checks_df = process_candidate_resumes(
                api_key_input, key_type.startswith("Vertex"), uploaded_files, blacklist_projects
            )

        st.success("Audit Complete!")

        st.subheader("Candidate Summary")
        st.dataframe(summary_df, use_container_width=True)
        st.download_button("📥 Download Summary (CSV)", summary_df.to_csv(index=False).encode("utf-8"),
                           file_name="screen_summary.csv", mime="text/csv")

        if not checks_df.empty:
            st.subheader("Check Details")
            st.dataframe(checks_df, use_container_width=True)
            st.download_button("📥 Download Check Details (CSV)", checks_df.to_csv(index=False).encode("utf-8"),
                               file_name="screen_check_details.csv", mime="text/csv")