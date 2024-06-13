# STEP 1
from fastapi import FastAPI, Form
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

# STEP 2
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")

ner = pipeline("ner", model=model, tokenizer=tokenizer)

app = FastAPI()

# STEP 3
@app.post("/text/")
async def text(text: str = Form()):

#STEP 4
    results = ner(text)

# STEP 5
    print(results)
    return "results"