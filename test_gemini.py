import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

print("Initializing Gemini...", flush=True)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite",
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    max_retries=1,
    timeout=60
)

print("Sending request to Gemini...", flush=True)

try:
    response = llm.invoke("Reply with only: Gemini is working.")

    print("\nSUCCESS!")
    print(response.content)

except Exception as error:
    print("\nERROR:")
    print(type(error).__name__)
    print(str(error))