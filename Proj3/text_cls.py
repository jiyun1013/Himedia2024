# STEP 1. import modules
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForSequenceClassification

#STEP 2. create inference instance
classifier = pipeline("sentiment-analysis", model="snunlp/KR-FinBert-SC")

#STEP 3. prepare input data
text = "작년 대비 10% 하락세."

#STEP 4. infrence
result = classifier(text)

# 4-1. preprocessing  (data -> tensor(blob))
# 4-2. inference  (tensor(blog) -> logit)
# 4-3. postprocssing  (logit -> data)

#STEP 5. visulize
print(result)