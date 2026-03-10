import streamlit as st
import os
import shutil
from agents.recruitment_agent import recruitment_agent
from langchain_core.messages import HumanMessage

def recruitment_page():
    st.title("🤝 AI Recruitment Agent")
    st.markdown("""
    Streamline your hiring process with AI. 
    1. Paste the **Job Description**.
    2. Upload **Resumes** (PDF/DOCX).
    3. Let the Agent **Analyze, Score, and Automate** the workflow.
    """)

    st.divider()

    # Inputs
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📋 Job Description")
        job_desc = st.text_area("Paste the job requirements here...", height=300)

    with col2:
        st.subheader("📄 Upload Resumes")
        uploaded_resumes = st.file_uploader(
            "Upload multiple resumes (PDF or DOCX)", 
            type=["pdf", "docx"], 
            accept_multiple_files=True
        )
        
        st.divider()
        st.subheader("🎯 Selection Threshold")
        st.markdown("Set the minimum score required to shortlist a candidate.")
        threshold = st.slider("Min %", min_value=0, max_value=100, value=75, help="Candidates scoring below this will be automatically rejected.")

    if st.button("🚀 Start Recruitment Workflow", type="primary", use_container_width=True):
        if not job_desc:
            st.error("Please provide a job description.")
        elif not uploaded_resumes:
            st.error("Please upload at least one resume.")
        else:
            with st.status("Running recruitment workflow...", expanded=True) as status:
                # 1. Save uploaded files to a temp directory
                temp_dir = "temp_resumes"
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                os.makedirs(temp_dir)

                resume_paths = []
                for uploaded_file in uploaded_resumes:
                    # Sanitize filename: replace spaces and special chars with underscores
                    safe_name = "".join([c if c.isalnum() or c in "._-" else "_" for c in uploaded_file.name])
                    path = os.path.join(temp_dir, safe_name)
                    with open(path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    resume_paths.append(path)
                
                status.write(f"✅ Saved {len(resume_paths)} resumes (sanitized names).")

                # 2. Initialize State
                initial_state = {
                    "job_description": job_desc,
                    "resume_paths": resume_paths,
                    "threshold": float(threshold),
                    "messages": [HumanMessage(content="Start recruitment process")]
                }

                # 3. Run Graph
                try:
                    final_state = recruitment_agent.invoke(initial_state)
                    
                    status.update(label="✅ Recruitment Complete!", state="complete")

                    st.success("Recruitment workflow successfully executed.")

                    # 4. Display Results
                    st.divider()
                    st.header("📊 Recruiter Report")
                    st.markdown(final_state.get("report", "No report generated."))

                    st.divider()
                    st.subheader("👥 Candidate Details")
                    
                    for candidate in final_state.get("candidates", []):
                        with st.expander(f"{candidate.name} - Score: {candidate.score} ({candidate.status.upper()})"):
                            st.write(f"**Email:** {candidate.email}")
                            st.write(f"**Experience:** {candidate.experience} years")
                            st.write(f"**Skills:** {', '.join(candidate.skills)}")
                            st.write(f"**Summary:** {candidate.resume_summary}")
                            if candidate.zoom_link:
                                st.write(f"**Zoom Link:** [Join Meeting]({candidate.zoom_link})")
                                if candidate.zoom_status:
                                    st.caption(f"Zoom Status: {candidate.zoom_status}")
                            
                            if candidate.email_status:
                                if "Success" in candidate.email_status:
                                    st.success(f"📧 {candidate.email_status}")
                                elif "SIMULATION" in candidate.email_status:
                                    st.warning(f"🧪 {candidate.email_status}")
                                else:
                                    st.error(f"❌ {candidate.email_status}")

                except Exception as e:
                    status.update(label="❌ Error occurred", state="error")
                    st.error(f"Workflow failed: {str(e)}")
                finally:
                    # Clean up
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)

if __name__ == "__main__":
    recruitment_page()
