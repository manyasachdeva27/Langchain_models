from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

llm=HuggingFaceEndpoint(
    repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0", #facebook model chota version of llama model
    task="text-generation",
    max_new_tokens=100,
    do_sample=False,
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India?")
print(result.content) 

