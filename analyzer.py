import os
import json
from openai import OpenAI
from schemas import AnalysisResult
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure OpenRouter using the OpenAI SDK
client = OpenAI(
    base_url="https://api.tokenfactory.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY")
)

def run_ats_checks(resume_text: str) -> list[str]:
    """Basic rule-based ATS checks."""
    issues = []
    if "@" not in resume_text or "." not in resume_text:
        issues.append("Email address missing or improperly formatted.")
    if len(resume_text) < 300:
        issues.append("Resume text is very short. Ensure you have detailed experience bullets.")
    if "references available upon request" in resume_text.lower():
        issues.append("Remove 'References available upon request' - it wastes valuable space.")
    return issues

def analyze_resume(resume_text: str, jd_text: str, semantic_matches: list[str]) -> AnalysisResult:
    ats_issues = run_ats_checks(resume_text)
    
    # Dynamically generate the JSON schema instructions from the Pydantic model
    schema_instructions = json.dumps(AnalysisResult.model_json_schema(), indent=2)
    system_prompt = f"""
    You are an advanced AI Career Coach and ATS (Applicant Tracking System) simulator.
    Analyze the provided Resume against the Job Description (JD).
    
    Job Description:
    {jd_text}
    
    Resume:
    {resume_text}
    
    Semantic matches found in resume (contextual matches to JD):
    {semantic_matches}
    
    Rule-based ATS issues detected:
    {ats_issues}
    
    You must respond with a single, strictly valid JSON object that adheres to this exact TypeScript interface:
    
    interface AnalysisResult {{
      scores: {{
        overall_match: number; // 0-100
        ats: number; // 0-100
        semantic: number; // 0-100
        technical: number; // 0-100
        experience: number; // 0-100
        education: number; // 0-100
      }};
      skills: {{
        matched_hard_skills: string[];
        missing_hard_skills: string[];
        matched_soft_skills: string[];
        missing_soft_skills: string[];
      }};
      experience: Array<{{
        required_experience: string;
        candidate_has: boolean;
        evidence: string;
      }}>;
      projects: Array<{{
        title: string;
        impact: string;
        feedback: string;
      }}>;
      keyword_coverage: {{
        matched_keywords: string[];
        missing_keywords: string[];
        percentage: number; // 0-100
      }};
      resume_feedback: {{
        weak_bullet_points: Array<{{
          original: string;
          rewrite: string; // Use XYZ formula and metrics
          reason: string;
        }}>;
        ats_formatting_issues: string[]; // Include the rule-based issues provided earlier
        missing_sections: string[];
      }};
      strengths: string[];
      gaps: string[];
      priority_improvements: string[];
      job_fit: {{
        verdict: string; // e.g., "High Fit", "Medium Fit", "Low Fit"
        reasoning: string;
      }};
      summary: string;
      recommended_preparation: string[]; // e.g., interview topics to study
    }}
    
    Return ONLY the JSON object. No markdown backticks, no explanations outside the JSON.
    """
    
    response = client.chat.completions.create(
        model="nvidia/Nemotron-3-Nano-Omni", # or "openai/gpt-4o"
        response_format={"type": "json_object"}, # Enforce JSON
        max_tokens=8000,
        messages=[
            {"role": "system", "content": "You are a helpful AI Career Coach that outputs strictly valid JSON matching the requested schema."},
            {"role": "user", "content": system_prompt}
        ]
    )
    
    llm_output = response.choices[0].message.content
    
    try:
        data = json.loads(llm_output)
        
        # Ensure rule-based ATS issues are merged into the LLM's response
        if ats_issues:
            existing_issues = data.get("resume_feedback", {}).get("ats_formatting_issues", [])
            data["resume_feedback"]["ats_formatting_issues"] = list(set(existing_issues + ats_issues))
            
        # Validate with Pydantic
        return AnalysisResult(**data)
    except Exception as e:
        print(f"Error parsing LLM JSON: {e}\nRaw Output: {llm_output}")
        raise ValueError(f"Failed to parse LLM response: {e}")
