# STEP 1
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline

# STEP 2
tokenizer = AutoTokenizer.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")
model = AutoModelForTokenClassification.from_pretrained("Leo97/KoELECTRA-small-v3-modu-ner")

ner = pipeline("ner", model=model, tokenizer=tokenizer)

# STEP 3
example = "서울역으로 안내해줘."

#STEP 4
ner_results = ner(example)

# STEP 5
print(ner_results)