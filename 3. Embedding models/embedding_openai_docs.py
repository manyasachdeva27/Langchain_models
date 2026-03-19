# multiple documents se embedding generate karna h

from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

embeddings=OpenAIEmbeddings(model="text-embedding-3-large",dimensions=32)

docs=[
    "Delhi is the capital of India.",
    "Mumbai is the financial capital of India.",
    "Bangalore is the IT capital of India.",
    "Chennai is the cultural capital of India.",
    "Kolkata is the cultural capital of India.",
    "Hyderabad is the IT capital of India.",
    "Ahmedabad is the IT capital of India.",
]

result=embeddings.embed_documents(docs)
print(str(result))
