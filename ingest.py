import json
import psycopg2
from pgvector.psycopg2 import register_vector
from sentence_transformers import SentenceTransformer

# 1. Load a fast, local embedding model (outputs 384 dimensions)
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Connect to PostgreSQL
conn = psycopg2.connect(
    dbname='vectordb', 
    user='postgres', 
    password='admin', 
    host='localhost', 
    port=5432
)
conn.autocommit = True
cur = conn.cursor()

# 3. Enable the pgvector extension and register the data type
cur.execute('CREATE EXTENSION IF NOT EXISTS vector;')
register_vector(conn)

# 4. Define the schema (matching the 384 dimensions of our model)
cur.execute('DROP TABLE IF EXISTS knowledge_base;')
cur.execute('''
    CREATE TABLE knowledge_base (
        id bigserial PRIMARY KEY, 
        content text, 
        embedding vector(384)
    );
''')

# 5. Ingest your existing JSON data
print("Reading knowledge_base.json...")
with open('data/knowledge_base.json', 'r') as f:
    # Assuming your JSON is a list of dictionaries with a 'text' key
    documents = json.load(f) 

print(f"Embedding and inserting {len(documents)} records...")
for doc in documents:
    content = doc['text']
    # Convert text to a vector array
    embedding = model.encode(content).tolist() 
    
    # Insert into the database
    cur.execute(
        'INSERT INTO knowledge_base (content, embedding) VALUES (%s, %s)', 
        (content, embedding)
    )

print("Ingestion complete! Data is now in PostgreSQL.")
cur.close()
conn.close()