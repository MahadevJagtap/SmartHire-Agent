import os
import json
import logging
from typing import List, Dict, Any
from models.candidate_model import Candidate, RecruitmentReport
from tools.resume_parser_tool import resume_parser_tool
from tools.scoring_tool import candidate_scoring_tool
from tools.zoom_tool import zoom_meeting_tool
from tools.email_tool import email_sender_tool
from tools.report_generator_tool import report_generator_tool

logger = logging.getLogger(__name__)

class RecruitmentService:
    @staticmethod
    def _extract_json(content: str) -> Dict[str, Any]:
        """Robustly extract JSON from LLM response strings."""
        try:
            # Try direct parse first
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to find JSON block in markdown
            if "```json" in content:
                try:
                    json_str = content.split("```json")[1].split("```")[0].strip()
                    return json.loads(json_str)
                except:
                    pass
            
            # Try to find the first '{' and last '}'
            try:
                start = content.find('{')
                end = content.rfind('}')
                if start != -1 and end != -1:
                    json_str = content[start:end+1]
                    return json.loads(json_str)
            except:
                pass
                
        return {}

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
        data = RecruitmentService._extract_json(response.content)
        
        return {
            "role": data.get("role", "Unknown"),
            "required_skills": data.get("required_skills", []),
            "required_experience": data.get("required_experience", 0),
            "preferred_skills": data.get("preferred_skills", [])
        }

    @staticmethod
    def process_resumes(llm, resume_paths: List[str], job_requirements: Dict[str, Any]) -> List[Candidate]:
        """Process multiple resumes and return candidate objects."""
        candidates = []
        
        for path in resume_paths:
            try:
                raw_data_json = resume_parser_tool.invoke({"file_path": path})
                raw_data = json.loads(raw_data_json)
                
                if "error" in raw_data:
                    logger.error(f"Parser error for {os.path.basename(path)}: {raw_data['error']}")
                    continue

                prompt = f"""
                Extract candidate information from this resume text:
                {raw_data['raw_text']}
                
                Requirements Context: {json.dumps(job_requirements)}
                
                Return a JSON object with:
                - name: string
                - email: string
                - skills: list of strings
                - experience: number (years)
                - projects: list of strings
                - resume_summary: short string (2-3 sentences)
                """
                response = llm.invoke(prompt)
                extracted_data = RecruitmentService._extract_json(response.content)
                
                # Create candidate even if some fields are missing (fallback to dummy values)
                candidate = Candidate(
                    candidate_id=os.path.basename(path),
                    name=extracted_data.get("name") or extracted_data.get("full_name") or f"Candidate_{len(candidates)+1}",
                    email=extracted_data.get("email", "N/A"),
                    skills=extracted_data.get("skills", []),
                    experience=float(extracted_data.get("experience", 0.0)),
                    projects=extracted_data.get("projects", []),
                    resume_summary=extracted_data.get("resume_summary", "No summary extracted."),
                    status="pending"
                )
                candidates.append(candidate)
            except Exception as e:
                logger.error(f"Failed to process resume {os.path.basename(path)}: {str(e)}")
                continue
                
        return candidates

    @staticmethod
    def score_candidate(llm, candidate: Candidate, job_requirements: Dict[str, Any]) -> Candidate:
        """Score a single candidate against requirements using LLM and scoring tool."""
        prompt = f"""
        Evaluate candidate '{candidate.name}' against these requirements:
        {json.dumps(job_requirements)}
        
        Candidate Details:
        - Skills: {candidate.skills}
        - Experience: {candidate.experience} years
        - Projects: {candidate.projects}
        - Summary: {candidate.resume_summary}
        
        Provide scores (0-100) for:
        - skills_match_score
        - experience_score
        - projects_score
        
        Return ONLY valid JSON.
        """
        try:
            response = llm.invoke(prompt)
            scores = RecruitmentService._extract_json(response.content)
            
            # Use the tool for final calculation
            eval_result_json = candidate_scoring_tool.invoke({
                "skills_match_score": float(scores.get("skills_match_score", 0)),
                "experience_score": float(scores.get("experience_score", 0)),
                "projects_score": float(scores.get("projects_score", 0))
            })
            eval_result = json.loads(eval_result_json)
            
            candidate.score = float(eval_result.get("final_score", 0.0))
            candidate.status = eval_result.get("status", "rejected")
        except Exception as e:
            logger.error(f"Scoring failed for {candidate.name}: {e}")
            candidate.score = 0.0
            candidate.status = "rejected"
            
        return candidate
