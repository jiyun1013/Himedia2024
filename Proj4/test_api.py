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
        # Read the contents of the uploaded file
        contents = await file.read()
        # Decode the text file contents from UTF-8 format
        text = contents.decode("utf-8")
        
        # Split the text into lines
        lines = text.split("\n")
        results = {}
        for line in lines:
            # Split each line into [이름] [시간] 내용
            parts = line.split(" ", 2)
            if len(parts) == 3:
                speaker = parts[0][1:-1]  # Remove "[" and "]" from the speaker's name
                content = parts[2]
                # Perform sentiment analysis for each line
                result = classifier(content.strip())[0]
                # Store the sentiment analysis result for each speaker
                if speaker not in results:
                    results[speaker] = []
                results[speaker].append({"label": result["label"], "score": result["score"]})
        
        # Return the sentiment analysis results for each speaker
        return {"results": results}
    else:
        # Raise an HTTP 400 error if the file is not a text file
        raise HTTPException(status_code=400, detail="Only text files (.txt) are allowed.")
