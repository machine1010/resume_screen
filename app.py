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
You are an expert technical recruiter and resume authenticity reviewer, screening candidates for an AI Engineer role.
Analyze the candidate's resume PDF. Use web lookups only to verify employers and tool release dates.
Today's date is {current_date}.

ROLE SUMMARY
6-8 years of application development with a focus on AI. Builds internal AI tools and workflows (Claude, Cursor)
for finance/accounting teams, integrated with Snowflake, Slack and Google Drive.

RULES
- Base every finding on the resume. If something is missing, write "NOT_STATED". Do not guess.
- Do not use age, gender, photo, nationality or college name in your judgment.
- Mark each check as GREEN, AMBER or RED.

AUTHENTICITY CHECKS
1. Contact Info: Extract email and phone and validate their format. Both missing = RED, one missing = AMBER.
2. Career Gaps: List gaps > 6 months. Unexplained gap in the last 3 years = AMBER (not a rejection reason on its own).
3. Education & Work Overlap: Full-time degree overlapping a full-time job = RED. Part-time/distance programmes are fine.
4. Concurrent Employment: Two full-time jobs overlapping by more than 2 months = RED.
5. Title Escalation: Unrealistic jumps (e.g., fresher to Lead in under 2 years, "Head of AI" with under 2 years of AI work).
   One jump = AMBER, two or more = RED.
6. AI Timeline Check: Flag claims impossible by date - GenAI/LLM experience before Nov 2022, LangGraph before 2024,
   MCP before Nov 2024, Claude Code before 2025. One = AMBER, two or more = RED.
7. Employer Check: Quick web check that employers from the last 5 years exist. Not found = AMBER.
8. Blacklisted Projects: Compare listed projects against this blacklist:
    {blacklisted_projects_str}
   Any match = RED.

ROLE FIT CHECKS
9. Experience: Total full-time experience (overlaps counted once, internships excluded).
   6-8 years = GREEN, 4.5-5 or 8-10 years = AMBER, otherwise RED. Also extract relevant AI experience in years.
10. Recent Skills (last 1 year): A skill counts only if it appears in a project from the last 12 months, not just the skills list.
    Must-have: Python/TypeScript, LLM APIs (Claude, OpenAI, etc.), RAG, Function Calling / Structured Outputs,
    Agents / Workflow Orchestration, AI Evaluation & Guardrails.
    Nice-to-have: Snowflake / Slack / Google Drive integrations, Claude / Cursor, finance automation, enabling non-technical users.
    5-6 must-haves = GREEN, 3-4 = AMBER, 0-2 = RED.
11. Project Substance: Do AI projects describe what was built and the result (users, metrics, scale)?
    2 or more such projects = GREEN, 1 = AMBER, none / buzzwords only = RED.

RECOMMENDATION (apply the first rule that matches)
- REJECT: Blacklist match, or Experience RED, or Recent Skills RED, or AI Timeline RED.
- FLAG_FOR_AUDIT: Any other RED.
- PROCEED_WITH_CAUTION: 2 or more AMBERs.
- PASS: Everything else.

OUTPUT
Return only valid JSON in this structure:
{{
  "full_name": "",
  "email": "",
  "phone": "",
  "current_title": "",
  "current_employer": "",
  "notice_period": "",                  // "Immediate" if no current employer on the resume
  "total_experience_years": 0,          // number only
  "relevant_ai_experience_years": 0,    // number only
  "top_skills": [],
  "missing_must_haves": [],
  "achievements": [],                   // top 3, keep numbers as written
  "discrepancies": [],
  "checks": [{{"id": 1, "name": "", "status": "GREEN | AMBER | RED", "finding": ""}}],
  "interview_questions": [],            // 3 questions targeting the weakest claims
  "recommendation": "PASS | PROCEED_WITH_CAUTION | FLAG_FOR_AUDIT | REJECT",
  "reason": ""
}}
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
