from fastapi import FastAPI, UploadFile, File, HTTPException
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from typing import List

app = FastAPI()

# Load sentiment analysis model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    results = []
    for file in files:
        # Ensure that the uploaded file is a text file
        if file.filename.endswith(".txt"):
            contents = await file.read()
            text = contents.decode("utf-8")

            # Perform sentiment analysis
            result = classifier(text)

            # Print the input text and the analysis result
            print(text)
            print(result[0])

            results.append({"filename": file.filename, "sentiment": result})
        else:
            raise HTTPException(status_code=400, detail="Only text files (.txt) are allowed.")

    return {"results": results}
