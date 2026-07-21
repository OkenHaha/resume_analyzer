import uuid
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import parser
import vector_db
import analyzer
from schemas import AnalysisResult

app = FastAPI(title="Advanced Resume Analyzer API")

# Allow CORS for your frontend (e.g., Next.js on localhost:3000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Change to your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/analyze", response_model=AnalysisResult)
async def analyze_resume(
    file: UploadFile = File(...), 
    job_description: str = Form(...)
):
    """
    Upload a Resume PDF and a Job Description text.
    This endpoint will automatically extract the resume text, 
    store it in the Vector DB, and run the AI analysis.
    """
    # 1. Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    # 2. Read PDF bytes and extract text
    contents = await file.read()
    resume_text = parser.extract_text_from_pdf(contents)
    
    if not resume_text:
        raise HTTPException(status_code=400, detail="Could not extract text from PDF. Is it an image-based PDF?")
    
    # 3. Chunk text and store in ChromaDB
    resume_id = str(uuid.uuid4())
    chunks = parser.chunk_text(resume_text, max_length=500)
    vector_db.store_resume_chunks(resume_id, chunks)
    
    # 4. Get semantic matches from ChromaDB
    semantic_matches = vector_db.query_semantic_matches(resume_id, job_description)
    
    # 5. Run LLM Analysis via OpenRouter
    try:
        result = analyzer.analyze_resume(resume_text, job_description, semantic_matches)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)