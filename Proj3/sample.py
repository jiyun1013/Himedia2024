# STEP 1. import modules
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification


#STEP 2. create inference instance
tokenizer = AutoTokenizer.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")
model = AutoModelForSequenceClassification.from_pretrained("nlp04/korean_sentiment_analysis_kcelectra")

classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

#STEP 3. prepare input data
text = "심심해"

#STEP 4. infrence
result = classifier(text)

# 4-1. preprocessing  (data -> tensor(blob))
# 4-2. inference  (tensor(blog) -> logit)
# 4-3. postprocssing  (logit -> data)

#STEP 5. visulize
print(result)