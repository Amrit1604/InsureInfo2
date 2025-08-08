"""
🎯 MINIMAL POLICY API - FOCUSED DOCUMENT ANALYSIS
=================================================
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List
import json
import time
import logging
import traceback
from datetime import datetime

# Import streamlined analyzer
from streamlined_analyzer import StreamlinedPolicyAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="🏥 Minimal Policy Analysis API",
    description="Fast, focused insurance policy analysis",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global analyzer
analyzer = None

# Request Models
class QueryRequest(BaseModel):
    documents: str = Field(..., description="URL of policy document")
    questions: List[str] = Field(..., description="List of insurance claim queries")

class HackrxResponse(BaseModel):
    answers: List[str]

@app.on_event("startup")
async def startup_event():
    """Initialize the analyzer"""
    global analyzer
    try:
        logger.info("🚀 Initializing streamlined analyzer...")
        analyzer = StreamlinedPolicyAnalyzer()
        logger.info("✅ Analyzer ready!")
    except Exception as e:
        logger.error(f"❌ Failed to initialize analyzer: {str(e)}")

@app.get("/")
async def root():
    return {"message": "🏥 Minimal Policy Analysis API", "status": "ready"}

@app.get("/health")
async def health_check():
    return {"status": "healthy", "analyzer_ready": analyzer is not None}

@app.post("/hackrx/run", response_model=HackrxResponse)
async def hackrx_run(request: QueryRequest):
    """Main hackathon endpoint with focused document analysis"""
    start_time = time.time()

    try:
        logger.info(f"📥 Processing {len(request.questions)} questions")

        if analyzer is None:
            raise HTTPException(status_code=500, detail="Analyzer not initialized")

        # Load the specific document
        logger.info(f"📄 Loading document: {request.documents[:60]}...")
        load_start = time.time()
        success = analyzer.load_document(request.documents)
        load_time = time.time() - load_start

        if not success:
            logger.error("❌ Document loading failed")
            # Return error responses
            answers = []
            for question in request.questions:
                answers.append("Document loading failed. Please ensure the policy document URL is valid and accessible.")
            return HackrxResponse(answers=answers)

        logger.info(f"✅ Document loaded in {load_time:.2f}s")

        # Process each question
        answers = []
        for i, question in enumerate(request.questions, 1):
            question_start = time.time()

            # Check time limit
            elapsed = time.time() - start_time
            if elapsed > 20:  # 20s time limit
                answer = "Processing time exceeded. Please contact customer service for detailed analysis."
                answers.append(answer)
                continue

            logger.info(f"🔍 Question {i}: {question[:60]}...")

            # Analyze with document
            answer = analyzer.analyze_question(question)
            answers.append(answer)

            question_time = time.time() - question_start
            logger.info(f"✅ Question {i} completed in {question_time:.2f}s")

            # Log the answer for debugging
            logger.info(f"📝 Answer {i}: {answer[:100]}...")

        total_time = time.time() - start_time
        logger.info(f"🎉 All questions processed in {total_time:.2f}s")

        # Check 25s compliance
        if total_time > 25.0:
            logger.warning(f"⚠️ Response exceeded 25s target: {total_time:.1f}s")
        else:
            logger.info(f"✅ 25s compliance achieved: {total_time:.1f}s")

        return HackrxResponse(answers=answers)

    except Exception as e:
        logger.error(f"❌ Error in hackrx_run: {str(e)}")
        logger.error(traceback.format_exc())

        raise HTTPException(
            status_code=500,
            detail={
                "error": "Internal server error",
                "message": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Minimal Policy Analysis API...")
    print("📋 Endpoint: POST http://localhost:8080/hackrx/run")
    print("🎯 Focused on document analysis only!")

    uvicorn.run("minimal_api:app", host="0.0.0.0", port=8080, reload=False)
