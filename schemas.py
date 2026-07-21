from pydantic import BaseModel, Field
from typing import List, Optional

class Scores(BaseModel):
    overall_match: int = Field(description="Overall fit score 0-100")
    ats: int = Field(description="ATS readability score 0-100")
    semantic: int = Field(description="Contextual match score 0-100")
    technical: int = Field(description="Technical skills match score 0-100")
    experience: int = Field(description="Experience level match score 0-100")
    education: int = Field(description="Education requirements match score 0-100")

class Skills(BaseModel):
    matched_hard_skills: List[str] = []
    missing_hard_skills: List[str] = []
    matched_soft_skills: List[str] = []
    missing_soft_skills: List[str] = []

class ExperienceMatch(BaseModel):
    required_experience: str
    candidate_has: bool
    evidence: str

class ProjectFeedback(BaseModel):
    title: str
    impact: str
    feedback: str

class KeywordCoverage(BaseModel):
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []
    percentage: int

class WeakBullet(BaseModel):
    original: str
    rewrite: str
    reason: str

class ResumeFeedback(BaseModel):
    weak_bullet_points: List[WeakBullet] = []
    ats_formatting_issues: List[str] = []
    missing_sections: List[str] = []

class JobFit(BaseModel):
    verdict: str
    reasoning: str

class AnalysisResult(BaseModel):
    scores: Scores
    skills: Skills
    experience: List[ExperienceMatch] = []
    projects: List[ProjectFeedback] = []
    keyword_coverage: KeywordCoverage
    resume_feedback: ResumeFeedback
    strengths: List[str] = []
    gaps: List[str] = []
    priority_improvements: List[str] = []
    job_fit: JobFit
    summary: str
    recommended_preparation: List[str] = []