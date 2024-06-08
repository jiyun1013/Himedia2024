# STEP 1. 임포트하기
from fastapi import FastAPI, Form
from transformers import pipeline

# STEP 2. 데이터 추론기 만들기
summarizer = pipeline("summarization", model="stevhliu/my_awesome_billsum_model")

app = FastAPI()

# STEP 3. 데이터 넣기
@app.post("/text/")
async def text(text: str = Form()):

# STEP 4. 데이터 추론하기
    result = summarizer(text)

# STEP 5. 보여주기
    return {"result" : result}