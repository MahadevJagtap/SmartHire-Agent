from typing import List, Dict
from langchain_core.tools import tool
import json

@tool
def candidate_scoring_tool(skills_match_score: float, experience_score: float, projects_score: float) -> str:
    """
    Calculates the final candidate score based on:
    Skill Match: 50%
    Experience: 30%
    Projects / Achievements: 20%
    
    Returns structured evaluation with final score.
    """
    # Normalize inputs to 0-100 if they aren't already
    total_score = (skills_match_score * 0.5) + (experience_score * 0.3) + (projects_score * 0.2)
    
    status = "shortlisted" if total_score >= 75 else "rejected"
    
    evaluation = {
        "skills_score": skills_match_score,
        "experience_score": experience_score,
        "projects_score": projects_score,
        "final_score": round(total_score, 2),
        "status": status
    }
    
    return json.dumps(evaluation)
