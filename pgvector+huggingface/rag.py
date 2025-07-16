import torch
from transformers import pipeline
import psycopg2
from sentence_transformers import SentenceTransformer


query = """In the master bedroom, what is the bed made out of?"""


model = SentenceTransformer('all-MiniLM-L6-v2')
embedding = model.encode([query])[0].tolist()

conn = psycopg2.connect(
    host="localhost",
    database="vector_db",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()
cur.execute(f"SELECT id, content FROM documents ORDER BY embedding <-> '{embedding}' LIMIT 1;")
conn.commit()
results = cur.fetchall()
cur.close()
conn.close()


prompt = f"""Answer the following question based on the context below.

Context:
{results[0][1]}

Question:
{query}

Answer:"""


pipe = pipeline(
    "text-generation", 
    model="meta-llama/Llama-3.2-3B", 
    torch_dtype=torch.bfloat16, 
    device_map="auto"
)

output = pipe(prompt, num_return_sequences=1)

print('\n'*5)
print(output[0]['generated_text'])
