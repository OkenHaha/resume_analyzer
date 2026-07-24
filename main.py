import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parser
import vector_db
import analyzer
from schemas import AnalysisResult
import json
from typing import Optional

app = FastAPI(title="Advanced Resume Analyzer API")

# Allow CORS for your frontend (e.g., Next.js on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=List[AnalysisResult]) # Changed to List[AnalysisResult]
async def analyze_resume(
    file: UploadFile = File(...), 
    job_description: Optional[str] = Form(None),
    job_json_file: Optional[UploadFile] = File(None)
):
    if not job_description and not job_json_file:
        raise HTTPException(status_code=400, detail="Please provide either 'job_description' as text or 'job_json_file' as a file.")

    # 1. Read PDF bytes and extract text ONCE (so we don't process the resume 11 times)
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    contents = await file.read()
    resume_text = parser.extract_text_from_pdf(contents)
    
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF.")
    
    # 2. Chunk and store resume in ChromaDB ONCE
    resume_id = str(uuid.uuid4())
    chunks = parser.chunk_text(resume_text, max_length=500)
    vector_db.store_resume_chunks(resume_id, chunks)

    # Prepare list of jobs to analyze
    jobs_to_analyze = []

    if job_json_file:
        json_contents = await job_json_file.read()
        try:
            parsed_data = json.loads(json_contents)
            
            # If it's a list, use the whole list. If it's a dict, wrap it in a list.
            if isinstance(parsed_data, list):
                jobs_to_analyze = parsed_data
            else:
                jobs_to_analyze = [parsed_data]
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON format.")
    else:
        # If using text, create a dummy dictionary so the loop works
        jobs_to_analyze = [{"job_title": "Manual Input", "about_the_job": job_description, "minimum_qualifications": [], "preferred_qualifications": [], "responsibilities": []}]

    results = []

    # 3. Loop through each job and run the analysis
    for job_data in jobs_to_analyze:
        job_title = job_data.get('job_title', 'Unknown Title')
        
        def format_list(items):
            if not items: return "Not specified"
            return "\n".join(f"- {item}" for item in items)
        
        final_job_description = f"""
        Job Title: {job_title}
        
        About the Job:
        {job_data.get('about_the_job', 'Not provided')}
        
        Minimum Qualifications:
        {format_list(job_data.get('minimum_qualifications', []))}
        
        Preferred Qualifications:
        {format_list(job_data.get('preferred_qualifications', []))}
        
        Responsibilities:
        {format_list(job_data.get('responsibilities', []))}
        """
        
        # Get semantic matches for THIS specific job
        semantic_matches = vector_db.query_semantic_matches(resume_id, final_job_description)
        
        try:
            # Run LLM Analysis
            result = analyzer.analyze_resume(resume_text, final_job_description, semantic_matches)
            results.append(result)
        except Exception as e:
            # If one job fails, we still want to return the successful ones
            print(f"Failed to analyze {job_title}: {e}")

    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)