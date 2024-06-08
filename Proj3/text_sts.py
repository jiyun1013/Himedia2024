# STEP 1
from sentence_transformers import SentenceTransformer

# STEP 2
model = SentenceTransformer("all-MiniLM-L6-v2")

# STEP 3
sentence1 = "The weather is lovely today."
sentence2 = "It's so sunny outside!"

# STEP 4 각각 임베딩하기
embedding1 = model.encode(sentence1)
embedding2 = model.encode(sentence2)

print(embedding1.shape)
print(embedding2.shape)

# [3, 384]

# STEP 5
similarities = model.similarity(embedding1, embedding2)
print(similarities)
# tensor([[1.0000, 0.6660, 0.1046],
#         [0.6660, 1.0000, 0.1411],
#         [0.1046, 0.1411, 1.0000]])