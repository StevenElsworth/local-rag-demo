import os
import psycopg2
from sentence_transformers import SentenceTransformer

conn = psycopg2.connect(
    host="localhost",
    database="vector_db",
    user="postgres",
    password="postgres"
)

cur = conn.cursor()

model = SentenceTransformer('all-MiniLM-L6-v2')

# Upload files from a directory
directory = "./data"
for filename in os.listdir(directory):
    with open(os.path.join(directory, filename), 'r') as f:
        content = f.read()
        embedding = model.encode(content).tolist()
	
        cur.execute(
            "INSERT INTO documents (content, embedding) VALUES (%s, %s)",
            (content, embedding)
        )
        
conn.commit()
cur.close()
conn.close()
