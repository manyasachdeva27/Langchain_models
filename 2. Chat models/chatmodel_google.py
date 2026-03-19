# saare ke saare chat models base base chat model se inherit hote h 

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()

#temperature -> 0 to 2 creativity parameter for deteministic task be more on 0 side , for creative task be more o 1.5 side
#max_completion_tokens-> output me kitne tokens (like words) chayiye
chatmodel=ChatGoogleGenerativeAI(model='gemini-1.5-pro',temperature=0.3, max_completion_tokens=100)

result=chatmodel.invoke("What is the capital of India?")

#print(result) ye metadata return krta h

print(result.content) #fetches only the answer

