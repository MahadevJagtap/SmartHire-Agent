import os
import json
from typing import List, Dict, Any
from models.candidate_model import Candidate, RecruitmentReport
from tools.resume_parser_tool import resume_parser_tool
from tools.scoring_tool import candidate_scoring_tool
from tools.zoom_tool import zoom_meeting_tool
from tools.email_tool import email_sender_tool
from tools.report_generator_tool import report_generator_tool

class RecruitmentService:
    @staticmethod
    def parse_job_requirements(llm, requirements_text: str) -> Dict[str, Any]:
        """Use LLM to extract structured requirements."""
        prompt = f"""
        Extract key job requirements from the following text:
        {requirements_text}
        
        Return a JSON object with:
        - role: string
        - required_skills: list of strings
        - required_experience: number
        - preferred_skills: list of strings
        """
        response = llm.invoke(prompt)
        # Assuming structured output or parsing JSON from response
        try:
            # Basic parsing if LLM doesn't return pure JSON
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {"role": "Unknown", "required_skills": [], "required_experience": 0, "preferred_skills": []}

    @staticmethod
    def process_resumes(llm, resume_paths: List[str], job_requirements: Dict[str, Any]) -> List[Candidate]:
        """Process multiple resumes and return candidate objects."""
        candidates = []
        for path in resume_paths:
            raw_data_json = resume_parser_tool.invoke({"file_path": path})
            raw_data = json.loads(raw_data_json)
            
            if "error" in raw_data:
                continue

            # Use LLM to extract candidate details from raw text based on requirements
            prompt = f"""
            Extract candidate information from this resume text:
            {raw_data['raw_text']}
            
            Based on these job requirements:
            {json.dumps(job_requirements)}
            
            Return a JSON object with:
            - name: string
            - email: string
            - skills: list of strings
            - experience: number (years)
            - projects: list of strings
            - resume_summary: short string (2-3 sentences)
            """
            response = llm.invoke(prompt)
            try:
                content = response.content
                if "```json" in content:
                    content = content.split("```json")[1].split("```")[0].strip()
                extracted_data = json.loads(content)
                
                candidate = Candidate(
                    candidate_id=os.path.basename(path),
                    name=extracted_data.get("name", "Unknown"),
                    email=extracted_data.get("email", ""),
                    skills=extracted_data.get("skills", []),
                    experience=extracted_data.get("experience", 0.0),
                    projects=extracted_data.get("projects", []),
                    resume_summary=extracted_data.get("resume_summary", ""),
                    status="pending"
                )
                candidates.append(candidate)
            except:
                continue
        return candidates

    @staticmethod
    def score_candidate(llm, candidate: Candidate, job_requirements: Dict[str, Any]) -> Candidate:
        """Score a single candidate against requirements using LLM and scoring tool."""
        # Use LLM to calculate sub-scores
        prompt = f"""
        Evaluate candidate '{candidate.name}' against these requirements:
        {json.dumps(job_requirements)}
        
        Candidate Skills: {candidate.skills}
        Candidate Experience: {candidate.experience} years
        Candidate Projects: {candidate.projects}
        
        Provide scores (0-100) for:
        - skills_match_score
        - experience_score
        - projects_score
        
        Return only JSON.
        """
        response = llm.invoke(prompt)
        try:
            content = response.content
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            scores = json.loads(content)
            
            # Use the tool for final calculation
            eval_result_json = candidate_scoring_tool.invoke({
                "skills_match_score": scores.get("skills_match_score", 0),
                "experience_score": scores.get("experience_score", 0),
                "projects_score": scores.get("projects_score", 0)
            })
            eval_result = json.loads(eval_result_json)
            
            candidate.score = eval_result.get("final_score", 0.0)
            candidate.status = eval_result.get("status", "rejected")
        except:
            candidate.score = 0.0
            candidate.status = "rejected"
            
        return candidate
