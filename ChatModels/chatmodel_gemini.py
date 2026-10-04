
from dotenv import load_dotenv
import logging
from langchain_google_genai import ChatGoogleGenerativeAI

# Hide the Google GenAI AFC warning
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-3.8-flash"
)  

response = llm.invoke("Greet me.")

# Extract the actual message
if isinstance(response.content, str):
    message = response.content
else:
    message = "\n".join(
        block["text"]
        for block in response.content
        if isinstance(block, dict) and block.get("type") == "text"
    )

# Extract token usage
total_tokens = response.usage_metadata["total_tokens"]

print("Message:", message)
print("Total Tokens Used:", total_tokens)

