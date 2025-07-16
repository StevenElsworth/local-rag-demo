import os
import psycopg2
from sentence_transformers import SentenceTransformer



model = SentenceTransformer('all-MiniLM-L6-v2')


sentences = [
    "How long would it take me to mow my very large garden given all the trees?"
]

embeddings = model.encode(sentences)
print(embeddings[0].tolist())
