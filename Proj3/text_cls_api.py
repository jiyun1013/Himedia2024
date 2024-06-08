# STEP 1. import modules
from fastapi import FastAPI, Form
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")
app = FastAPI()

@app.post("/text/")
async def text(text: str = Form()):
    #STEP 3. prepare input data
# text = "작년 대비 10% 하락세."

# STEP 4.
# inputs = tokenizer(text, return_tensors="pt")
# with torch.no_grad():
#     logits = model(**inputs).logits

#STEP 4. infrence
    result = classifier(text)

# 4-1. preprocessing  (data -> tensor(blob))
# 4-2. inference  (tensor(blog) -> logit)
# 4-3. postprocssing  (logit -> data)

#STEP 5. visulize
    return {"result" : result}