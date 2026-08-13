import streamlit as st
import os
import tempfile
import pandas as pd
from typing import List
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

# ---------------------------------------------------------------------------
# 1. Pydantic Output Schema
# ---------------------------------------------------------------------------
class AuditCheckItem(BaseModel):
    check_name: str = Field(
        description="Name of the check executed (e.g., Experience Range Filter, Blacklisted Project Check, Recent Skills Check)."
    )
    outcome: str = Field(
        description="Detailed findings, evidence, date calculations, or detected blacklisted terms."
    )
    flag_status: str = Field(
        description="Flag level: PASS, YELLOW_FLAG, RED_FLAG, PROCEED, PROCEED_WITH_CAUTION, FLAG_FOR_AUDIT, or REJECT."
    )

class CandidateResumeAudit(BaseModel):
    candidate_name: str = Field(
        description="The full name of the candidate extracted from the resume."
    )
    checks: List[AuditCheckItem]

# ---------------------------------------------------------------------------
# 2. Base System Instruction Template
# ---------------------------------------------------------------------------
SYSTEM_INSTRUCTION_TEMPLATE = """
You are an expert HR Forensic Auditor and Resume Authenticity Specialist.
Analyze the candidate's resume PDF and execute live web lookups where necessary. Extract the candidate's full name.

AUDIT CHECKS TO EXECUTE:
1. Contact Info Verification (Email and Phone validation).
2. LinkedIn Profile Extraction & Live Web Verification (Connection count and profile match).
3. Career Gap Analysis (Identify gaps > 3 months for most recent Career Gap).
4. Education & Work Experience Overlap (Degrees overlapping full-time jobs).
5. Company-to-Company Tenure Overlap (Concurrent full-time employment).
6. Career Trajectory & Title Escalation (Unrealistic title jumps).
7. Recent Skills Application Check (Last 1 Year): Verify skills(snowflake and DBT(data build tool) ) appear in projects worked on in the last 1 year.
8. Experience Range Filter (5 to 8 Years): Total experience MUST be between 4 and 8 years. Flag as RED_FLAG if outside 5-8 years.
9. Blacklisted Project Audit: Compare listed projects against this blacklist:
    {blacklisted_projects_str}
    Flag as RED_FLAG if any match is found.
10. Overall Candidate Risk Score & Recommendation (PASS / PROCEED_WITH_CAUTION / FLAG_FOR_AUDIT / REJECT).
"""

# ---------------------------------------------------------------------------
# 3. Processing Function (Option 2: Vertex AI Direct Bytes Injection)
# ---------------------------------------------------------------------------
def process_candidate_resumes(api_key: str, uploaded_files_list: List, blacklisted_projects: List[str]) -> pd.DataFrame:
    # Initialize client for Vertex AI
    client = genai.Client(vertexai=True, api_key=api_key)

    # Format blacklist into bulleted string
    blacklist_formatted = "\n".join([f"- {proj}" for proj in blacklisted_projects])

    # Build complete prompt with dynamic blacklist
    system_instruction = SYSTEM_INSTRUCTION_TEMPLATE.format(
        blacklisted_projects_str=blacklist_formatted
    )

    all_report_rows = []

    for uploaded_file in uploaded_files_list:
        file_name = uploaded_file.name
        
        try:
            # Read raw binary data directly from Streamlit's UploadedFile object
            pdf_bytes = uploaded_file.getvalue()

            # Create a inline Part object using types.Part.from_bytes
            pdf_part = types.Part.from_bytes(
                data=pdf_bytes,
                mime_type="application/pdf"
            )

            # Generate content using inline PDF bytes
            response = client.models.generate_content(
                model="gemini-3.6-flash",
                contents=[
                    pdf_part,
                    f"Perform a full forensic audit on '{file_name}'. Verify total work experience is 4-8 years, scan for recent skills in 1-year projects, check against blacklisted projects, and execute all 11 audit checks."
                ],
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    response_mime_type="application/json",
                    response_schema=CandidateResumeAudit,
                ),
            )

            # Parse JSON output via Pydantic
            audit_result = CandidateResumeAudit.model_validate_json(response.text)

            for check in audit_result.checks:
                all_report_rows.append({
                    "Candidate Name": audit_result.candidate_name,
                    "File Name": file_name,
                    "Check Name": check.check_name,
                    "Findings / Outcome": check.outcome,
                    "Flag Status": check.flag_status
                })

        except Exception as e:
            all_report_rows.append({
                "Candidate Name": "Unknown",
                "File Name": file_name,
                "Check Name": "Processing Error",
                "Findings / Outcome": f"Failed to audit: {str(e)}",
                "Flag Status": "RED_FLAG"
            })

    return pd.DataFrame(all_report_rows)

# ---------------------------------------------------------------------------
# 4. Streamlit User Interface
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Resume Auditor", layout="wide")

st.title("🕵️‍♂️ HR Resume Scanner")
st.markdown("Upload candidate resumes (PDFs) to run a forensic 11-point audit and download the results as a CSV.")

# --- Sidebar Controls ---
with st.sidebar:
    st.header("Configuration")
    
    # Securely handle the API key
    api_key_input = st.text_input(
        "Vertex AI / Express API Key", 
        type="password", 
        help="Provide your authentication key."
    )
    
    # Editable Blacklist Projects
    st.subheader("Blacklisted Projects")
    default_blacklist = (
        "Project Titan - Online Banking System (Proxy Template)\n"
        "ABC Technologies E-Commerce Management System\n"
        "Global Logistics Tracking Hub (Training Scam)\n"
        "Healthcare Patient Portal v2.0 (Common Copy-Paste Project)"
    )
    blacklist_text = st.text_area("Enter one project per line:", value=default_blacklist, height=200)

# --- Main Area ---
uploaded_files = st.file_uploader("Upload PDF Resumes", type=["pdf"], accept_multiple_files=True)

if st.button("Run Audit"):
    if not api_key_input:
        st.error("Please provide an API Key in the sidebar.")
    elif not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        # Parse blacklist textarea into a Python list
        blacklist_projects = [line.strip() for line in blacklist_text.split("\n") if line.strip()]
        
        # Run Process using memory streams (no disk writing needed)
        with st.spinner(f"Auditing {len(uploaded_files)} resume(s)... This may take a minute."):
            result_df = process_candidate_resumes(api_key_input, uploaded_files, blacklist_projects)
        
        st.success("Audit Complete!")
        
        # Display preview
        st.subheader("Audit Results Preview")
        st.dataframe(result_df, use_container_width=True)
        
        # Convert DF to CSV and create a download button
        csv_data = result_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Full Report (CSV)",
            data=csv_data,
            file_name="screen_results.csv",
            mime="text/csv",
        )
