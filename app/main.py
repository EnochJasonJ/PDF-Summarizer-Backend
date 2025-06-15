import io
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from app.utils import extract_text_from_pdf, summarize_text

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://127.0.0.1:5500"] for more security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/summarize-pdf/")
async def summarize_pdf(file: UploadFile = File(...)):
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are supported."}
    
    contents = await file.read()
    text = extract_text_from_pdf(io.BytesIO(contents))
    
    if not text:
        return {"error": "No text found in PDF."}
    
    summary = summarize_text(text)
    return {"summary": summary}
