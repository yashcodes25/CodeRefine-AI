from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from groq import Groq
import os
from dotenv import load_dotenv
import re
from typing import List

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GROQ CLIENT ──
client = Groq(api_key=os.getenv("GROQ_API_KEY"))
MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# ── MODELS ──
class CodeReviewRequest(BaseModel):
    code: str
    language: str = "Python"
    focus_areas: List[str] = ["bugs", "security", "performance"]

class CodeReviewResponse(BaseModel):
    review: str
    language: str

# ── PARSE REVIEW ──
def parse_review(review_txt: str) -> dict:
    """Parse the LLM response to extract structured data"""
    critical_section = re.search(r'### Critical Issues.*?(?=###|\Z)', review_txt, re.DOTALL)
    high_section     = re.search(r'### High Priority Issues.*?(?=###|\Z)', review_txt, re.DOTALL)
    medium_section   = re.search(r'### Medium Priority Issues.*?(?=###|\Z)', review_txt, re.DOTALL)
    low_section      = re.search(r'### Low Priority Issues.*?(?=###|\Z)', review_txt, re.DOTALL)

    return {
        "critical": critical_section.group(0) if critical_section else "",
        "high":     high_section.group(0)     if high_section     else "",
        "medium":   medium_section.group(0)   if medium_section   else "",
        "low":      low_section.group(0)       if low_section      else "",
    }

# ── ROUTES ──

@app.get("/")
async def root():
    """Redirect root to /app"""
    return RedirectResponse(url="/app")

@app.get("/login", response_class=HTMLResponse)
async def serve_login():
    """Serve the login page"""
    try:
        with open("../frontend/login.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>login.html not found</h1>", status_code=404)

@app.get("/app", response_class=HTMLResponse)
async def serve_tool():
    """Serve the main code review page"""
    try:
        with open("../frontend/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="<h1>index.html not found</h1>", status_code=404)

@app.post("/app/review", response_model=CodeReviewResponse)
async def review_code(request: CodeReviewRequest):
    """Review code and provide suggestions using GROQ API"""
    if not request.code.strip():
        raise HTTPException(status_code=400, detail="Code cannot be empty")

    focus_str = ", ".join(request.focus_areas)

    prompt = f"""You are an expert code reviewer with 15+ years of experience. Analyze the following {request.language} code and provide a structured review.

Code to review:
```{request.language}
{request.code}
```

Focus areas: {focus_str}

Provide your review in the following format:

### Critical Issues
List any critical bugs, security vulnerabilities, or breaking errors. If none, write "None found."

### High Priority Issues
List significant problems affecting functionality or security. If none, write "None found."

### Medium Priority Issues
List code quality problems, anti-patterns, or maintainability concerns. If none, write "None found."

### Low Priority Issues
List minor style issues, naming conventions, or small improvements. If none, write "None found."

### Rewritten Code
Provide a fully corrected and optimized version of the code.

### Summary
Brief explanation of the key changes made and why.
"""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=2000,
        top_p=0.9,
    )

    review_text = response.choices[0].message.content

    return CodeReviewResponse(
        review=review_text,
        language=request.language
    )

# ── ENTRY POINT ──
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)