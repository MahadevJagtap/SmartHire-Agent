from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class Candidate(BaseModel):
    candidate_id: str = Field(..., description="Unique identifier for the candidate")
    name: str = Field(..., description="Full name of the candidate")
    email: str = Field(..., description="Email address of the candidate")
    skills: List[str] = Field(default_factory=list, description="List of skills extracted from resume")
    experience: float = Field(0.0, description="Years of experience")
    projects: List[str] = Field(default_factory=list, description="List of projects or achievements")
    resume_summary: str = Field("", description="Short summary of the candidate's resume")
    score: float = Field(0.0, description="Overall candidate score (0-100)")
    status: Literal["shortlisted", "rejected", "pending"] = Field("pending", description="Current status of the candidate")
    zoom_link: Optional[str] = Field(None, description="Zoom meeting link for interview")
    zoom_status: Optional[str] = Field(None, description="Status of Zoom meeting tool call")
    email_status: Optional[str] = Field(None, description="Status of email tool call")
    rejection_reason: Optional[str] = Field(None, description="Reason for rejection")

class RecruitmentReport(BaseModel):
    total_candidates: int
    shortlisted_count: int
    rejected_count: int
    top_candidates: List[Candidate]
    summary: str
