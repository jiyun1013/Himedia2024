from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List

app = FastAPI()

# Load sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    # Ensure that the uploaded file is a text file
    if file.filename.endswith(".txt"):
        contents = await file.read()
        text = contents.decode("utf-8")
        
        # Split the text into lines and analyze each line
        lines = text.split("\n")
        results = []
        for line in lines:
            # Perform sentiment analysis on each line
            result = classifier(line)
            results.append({"text": line, "sentiment": result})
        
        return {"results": results}
    else:
        raise HTTPException(status_code=400, detail="Only text files (.txt) are allowed.")
