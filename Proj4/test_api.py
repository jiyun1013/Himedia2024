# STEP 1. import modules
from fastapi import FastAPI, Form
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#STEP 2. create inference instance
tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)
app = FastAPI()

#STEP 3. prepare input data
@app.post("/text/")
async def text(text: str = Form()):

#STEP 4. infrence
    result = classifier(text)

#STEP 5. visulize
    print(text)
    print(result[0])
    return {"result" : result}