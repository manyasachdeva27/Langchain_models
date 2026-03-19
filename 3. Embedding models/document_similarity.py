from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

docs=[
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT capital of India.",
    "Chennai is the cultural capital of India.",
    "Kolkata is the cultural capital of India."
]

query="What is the capital of India?"

doc_embeddings=embeddings.embed_documents(docs)

query_embedding=embeddings.embed_query(query)

scores=cosine_similarity([query_embedding], doc_embeddings)[0]
index,score=sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[-1]

#get the index of the most similar document
print(query)
print(docs[index])
print(f"The most similar document is {docs[index]} with a similarity score of {score}")