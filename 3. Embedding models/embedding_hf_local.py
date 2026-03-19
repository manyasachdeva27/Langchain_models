from langchain_huggingface import HuggingFaceEmbeddings

embeddings=HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

#similarly you can embed multiple documents
text="Delhi is the capital of India."

embedding=embeddings.embed_query(text)
print(str(embedding))