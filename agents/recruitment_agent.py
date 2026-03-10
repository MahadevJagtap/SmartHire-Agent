import json
import operator
from typing import List, Dict, Any, Annotated, TypedDict
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_groq import ChatGroq
from models.candidate_model import Candidate, RecruitmentReport
from services.recruitment_service import RecruitmentService
from tools.zoom_tool import zoom_meeting_tool
from tools.email_tool import email_sender_tool
from tools.report_generator_tool import report_generator_tool

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
class RecruitmentState(TypedDict):
    job_description: str
    resume_paths: List[str]
    job_requirements: Dict[str, Any]
    candidates: List[Candidate]
    shortlisted_candidates: List[Candidate]
    rejected_candidates: List[Candidate]
    threshold: float
    report: str
    logs: Annotated[List[str], operator.add]
    messages: List[BaseMessage]

# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------
llm = ChatGroq(model="llama-3.1-8b-instant")

# ---------------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------------

def parse_job_requirements(state: RecruitmentState) -> Dict[str, Any]:
    print("--- PARSING JOB REQUIREMENTS ---")
    requirements = RecruitmentService.parse_job_requirements(llm, state["job_description"])
    return {"job_requirements": requirements}

def ingest_resumes(state: RecruitmentState) -> Dict[str, Any]:
    print("--- INGESTING RESUMES ---")
    # This node just confirms visibility of resume paths
    return {"resume_paths": state["resume_paths"]}

def resume_parser_node(state: RecruitmentState) -> Dict[str, Any]:
    print("--- PARSING RESUMES ---")
    logs = state.get("logs", [])
    candidates = RecruitmentService.process_resumes(llm, state["resume_paths"], state["job_requirements"], logs)
    return {"candidates": candidates, "logs": logs}

def generate_resume_summary(state: RecruitmentState) -> Dict[str, Any]:
    print("--- GENERATING RESUME SUMMARIES ---")
    # Already handled in recruitment_service.process_resumes
    return {}

def requirement_matching_node(state: RecruitmentState) -> Dict[str, Any]:
    print("--- MATCHING REQUIREMENTS ---")
    # Logic is part of scoring, but we could add explicit steps here if needed
    return {}

def candidate_scoring_node(state: RecruitmentState) -> Dict[str, Any]:
    print("--- SCORING CANDIDATES ---")
    scored_candidates = []
    for candidate in state["candidates"]:
        scored = RecruitmentService.score_candidate(llm, candidate, state["job_requirements"])
        scored_candidates.append(scored)
    return {"candidates": scored_candidates}

def shortlist_candidates(state: RecruitmentState) -> Dict[str, Any]:
    print("--- SHORTLISTING CANDIDATES ---")
    threshold = state.get("threshold", 75.0)
    # Re-evaluate status based on dynamic threshold
    shortlisted = []
    rejected = []
    for c in state["candidates"]:
        if c.score >= threshold:
            c.status = "shortlisted"
            shortlisted.append(c)
        else:
            c.status = "rejected"
            rejected.append(c)
    return {"shortlisted_candidates": shortlisted, "rejected_candidates": rejected, "candidates": state["candidates"]}

def generate_zoom_link(state: RecruitmentState) -> Dict[str, Any]:
    print("--- GENERATING ZOOM LINKS ---")
    for candidate in state["shortlisted_candidates"]:
        link = zoom_meeting_tool.invoke({"candidate_name": candidate.name})
        candidate.zoom_link = link
        # Extract status message from tool response if it's a simulation/error
        if " simulation/" in link or "api-error" in link or "exception" in link:
            candidate.zoom_status = f"Warning: {link.split('/')[-1].replace('-', ' ').title()}"
        else:
            candidate.zoom_status = "Success: Meeting generated"
    return {"shortlisted_candidates": state["shortlisted_candidates"]}

def send_interview_email(state: RecruitmentState) -> Dict[str, Any]:
    print("--- SENDING INTERVIEW EMAILS ---")
    role = state["job_requirements"].get("role", "the position")
    for candidate in state["shortlisted_candidates"]:
        subject = "Interview Invitation"
        body = f"Hello {candidate.name},\n\nYou have been shortlisted for the {role} position.\n\nInterview Meeting Link:\n{candidate.zoom_link}\n\nBest regards\nAI Recruitment Agent"
        status = email_sender_tool.invoke({"to_email": candidate.email, "subject": subject, "body": body})
        candidate.email_status = status
    return {}

def send_rejection_email(state: RecruitmentState) -> Dict[str, Any]:
    print("--- SENDING REJECTION EMAILS ---")
    role = state["job_requirements"].get("role", "the position")
    for candidate in state["rejected_candidates"]:
        subject = "Application Update"
        body = f"Hello {candidate.name},\n\nThank you for applying for the {role} position.\n\nAfter reviewing your application, we will not be proceeding further.\n\nWe appreciate your interest and encourage you to apply again in the future."
        status = email_sender_tool.invoke({"to_email": candidate.email, "subject": subject, "body": body})
        candidate.email_status = status
    return {}

def generate_recruitment_report(state: RecruitmentState) -> Dict[str, Any]:
    print("--- GENERATING RECRUITMENT REPORT ---")
    candidates_data = [c.model_dump() for c in state["candidates"]]
    report = report_generator_tool.invoke({"candidates_data": json.dumps(candidates_data)})
    return {"report": report}

# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

workflow = StateGraph(RecruitmentState)

workflow.add_node("parse_job_requirements", parse_job_requirements)
workflow.add_node("ingest_resumes", ingest_resumes)
workflow.add_node("resume_parser", resume_parser_node)
workflow.add_node("generate_resume_summary", generate_resume_summary)
workflow.add_node("requirement_matching", requirement_matching_node)
workflow.add_node("candidate_scoring", candidate_scoring_node)
workflow.add_node("shortlist_candidates", shortlist_candidates)
workflow.add_node("generate_zoom_link", generate_zoom_link)
workflow.add_node("send_interview_email", send_interview_email)
workflow.add_node("send_rejection_email", send_rejection_email)
workflow.add_node("generate_recruitment_report", generate_recruitment_report)

workflow.add_edge(START, "parse_job_requirements")
workflow.add_edge("parse_job_requirements", "ingest_resumes")
workflow.add_edge("ingest_resumes", "resume_parser")
workflow.add_edge("resume_parser", "generate_resume_summary")
workflow.add_edge("generate_resume_summary", "requirement_matching")
workflow.add_edge("requirement_matching", "candidate_scoring")
workflow.add_edge("candidate_scoring", "shortlist_candidates")
workflow.add_edge("shortlist_candidates", "generate_zoom_link")
workflow.add_edge("generate_zoom_link", "send_interview_email")
workflow.add_edge("send_interview_email", "send_rejection_email")
workflow.add_edge("send_rejection_email", "generate_recruitment_report")
workflow.add_edge("generate_recruitment_report", END)

recruitment_agent = workflow.compile()
