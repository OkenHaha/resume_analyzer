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
    # system_prompt= f"""
    # You are an expert ATS (Applicant Tracking System) simulator, technical recruiter, and AI career coach.

    # Your task is to evaluate how well the candidate's resume matches the provided Job Description (JD).

    # Base your analysis ONLY on the following inputs:

    # 1. Job Description
    # 2. Resume
    # 3. Semantic matches
    # 4. Rule-based ATS issues

    # Do NOT invent or assume any information that is not explicitly present in these inputs.

    # If a required piece of information cannot be verified, treat it as missing rather than guessing.

    # ----------------------------------------
    # INPUTS
    # ----------------------------------------

    # Job Description

    # {jd_text}

    # Resume

    # {resume_text}

    # Semantic Matches

    # {semantic_matches}

    # Rule-based ATS Issues

    # {ats_issues}

    # ----------------------------------------
    # ANALYSIS GUIDELINES
    # ----------------------------------------

    # Perform a comprehensive comparison between the Job Description and the Resume.

    # Evaluate the following:

    # 1. Technical skills
    # 2. Soft skills
    # 3. Relevant experience
    # 4. Education
    # 5. Projects
    # 6. Keyword coverage
    # 7. ATS compatibility
    # 8. Overall suitability

    # ----------------------------------------
    # SCORING
    # ----------------------------------------

    # Generate the following integer scores (0–100).

    # overall_match

    # Overall suitability considering every aspect of the resume.

    # ats

    # ATS friendliness considering formatting, section organization, keyword usage, readability, and the provided ATS issues.

    # semantic

    # Contextual similarity between the resume and the Job Description.

    # technical

    # Coverage of required technical skills.

    # experience

    # Alignment of professional experience, responsibilities, seniority, and domain knowledge.

    # education

    # Alignment of education, certifications, and academic qualifications.

    # Scores should be internally consistent.

    # For example:

    # • High technical score should generally correspond to many matched hard skills.
    # • Low ATS score should correspond to several formatting issues.
    # • Overall score should reasonably reflect the other category scores.

    # ----------------------------------------
    # SKILLS
    # ----------------------------------------

    # Extract skills from both the Job Description and Resume.

    # Return:

    # matched_hard_skills

    # Technical skills found in both.

    # missing_hard_skills

    # Technical skills required by the JD but not found in the resume.

    # matched_soft_skills

    # Soft skills supported by resume evidence.

    # missing_soft_skills

    # Soft skills requested in the JD but not demonstrated.

    # Avoid duplicate skills.

    # Treat synonymous technologies as one concept where appropriate.

    # Example:

    # REST API
    # RESTful APIs

    # should count as the same skill.

    # ----------------------------------------
    # EXPERIENCE
    # ----------------------------------------

    # Summarize how well the candidate satisfies the experience requirements.

    # Return an object describing:

    # • required experience
    # • candidate experience
    # • major strengths
    # • missing experience

    # Do not invent years of experience.

    # Use only evidence found in the resume.

    # ----------------------------------------
    # PROJECTS
    # ----------------------------------------

    # Identify projects relevant to the target role.

    # For each project include:

    # title

    # impact

    # feedback

    # Impact should summarize why the project is valuable.

    # Feedback should explain how the project could better demonstrate relevance to the Job Description.

    # Ignore unrelated projects.

    # ----------------------------------------
    # KEYWORD COVERAGE
    # ----------------------------------------

    # Extract the important keywords from the Job Description.

    # Return:

    # matched_keywords

    # Keywords found in the resume.

    # missing_keywords

    # Important keywords absent from the resume.

    # percentage

    # Estimate keyword coverage from 0–100.

    # ----------------------------------------
    # RESUME FEEDBACK
    # ----------------------------------------

    # Weak Bullet Points

    # Identify bullet points that:

    # • are vague
    # • use passive language
    # • lack measurable impact
    # • fail to communicate achievements

    # Rewrite each bullet using stronger action verbs.

    # Use the Google XYZ formula whenever possible.

    # Accomplished X

    # measured by Y

    # by doing Z

    # If numerical metrics do not exist in the resume, do NOT invent numbers.

    # Improve wording while preserving factual accuracy.

    # Reason should briefly explain why the original bullet is weak.

    # ATS Formatting Issues

    # Include every provided ATS issue.

    # Do not invent additional formatting issues unless they are directly observable.

    # Missing Sections

    # Only include commonly expected resume sections such as:

    # Professional Summary

    # Skills

    # Projects

    # Certifications

    # Achievements

    # Portfolio

    # Volunteer Experience

    # Links

    # Only report sections that are actually missing and would strengthen the resume.

    # ----------------------------------------
    # STRENGTHS
    # ----------------------------------------

    # List the candidate's strongest qualifications supported by the resume.

    # Focus on:

    # • technical expertise
    # • relevant experience
    # • education
    # • projects
    # • achievements

    # ----------------------------------------
    # GAPS
    # ----------------------------------------

    # Identify the biggest weaknesses preventing a stronger match.

    # Examples include:

    # Missing technologies

    # Missing certifications

    # Missing domain experience

    # Missing leadership

    # Missing cloud experience

    # Only include evidence-based gaps.

    # ----------------------------------------
    # PRIORITY IMPROVEMENTS
    # ----------------------------------------

    # Return the five improvements that would most increase the candidate's suitability for this role.

    # Order them from highest impact to lowest impact.

    # Recommendations should be practical and specific.

    # ----------------------------------------
    # JOB FIT
    # ----------------------------------------

    # Determine the candidate's overall suitability.

    # Use one of:

    # Excellent Fit

    # High Fit

    # Medium Fit

    # Low Fit

    # Poor Fit

    # Explain the reasoning briefly.

    # ----------------------------------------
    # SUMMARY
    # ----------------------------------------

    # Write a concise 2–4 sentence summary describing:

    # • overall strengths

    # • biggest weaknesses

    # • overall recommendation

    # Do not mention numeric scores.

    # ----------------------------------------
    # RECOMMENDED PREPARATION
    # ----------------------------------------

    # Suggest interview topics or skills the candidate should review before interviewing.

    # Prioritize weak areas identified during the analysis.

    # ----------------------------------------
    # OUTPUT REQUIREMENTS
    # ----------------------------------------

    # Return ONLY one valid JSON object.

    # Do NOT include Markdown.

    # Do NOT include explanations.

    # Do NOT include comments.

    # Do NOT include trailing commas.

    # Every field in the schema must be present.

    # Use empty arrays [] where appropriate.

    # Use empty objects {} where appropriate.

    # Return valid JSON only.

    # The JSON MUST exactly follow this structure:

    # {
    #   "scores": {
    #     "overall_match": number,
    #     "ats": number,
    #     "semantic": number,
    #     "technical": number,
    #     "experience": number,
    #     "education": number
    #   },
    #   "skills": {
    #     "matched_hard_skills": [],
    #     "missing_hard_skills": [],
    #     "matched_soft_skills": [],
    #     "missing_soft_skills": []
    #   },
    #   "experience": {},
    #   "projects": [],
    #   "keyword_coverage": {},
    #   "resume_feedback": {
    #     "weak_bullet_points": [],
    #     "ats_formatting_issues": [],
    #     "missing_sections": []
    #   },
    #   "strengths": [],
    #   "gaps": [],
    #   "priority_improvements": [],
    #   "job_fit": {},
    #   "summary": "",
    #   "recommended_preparation": []
    # }
    # """
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
