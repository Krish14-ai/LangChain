from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv

load_dotenv()

model = ChatAnthropic(
    model_name="claude-haiku-4-5",
    temperature=1
)

result =model.invoke("HI")

print(result.content)