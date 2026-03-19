from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os
# model_id=repo_id
os.environ["HUGGINGFACE_HOME"]="D:/huggingface_cache"

llm=HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 100, "do_sample": False}
)

model=ChatHuggingFace(llm=llm)

result=model.invoke("What is the capital of India?")
print(result.content) 

