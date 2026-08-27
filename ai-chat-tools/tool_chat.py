import os
from datetime import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

# 1. The real Python function Claude is allowed to call
def get_current_time():
    return datetime.now().strftime("%A, %B %d, %Y at %I:%M %p")

# 2. Describe the tool so Claude knows it exists
tools = [{
    "name": "get_current_time",
    "description": "Returns the current date and time.",
    "input_schema": {"type": "object", "properties": {}}
}]

messages = [{"role": "user", "content": "What time is it right now?"}]

# 3. First call — Claude may ask to use the tool
response = client.messages.create(
    model="claude-sonnet-4-5", max_tokens=300, tools=tools, messages=messages
)

# 4. If Claude asked for the tool, run it and send the result back
if response.stop_reason == "tool_use":
    tool_call = next(b for b in response.content if b.type == "tool_use")
    result = get_current_time()
    messages.append({"role": "assistant", "content": response.content})
    messages.append({"role": "user", "content": [{
        "type": "tool_result",
        "tool_use_id": tool_call.id,
        "content": result
    }]})
    final = client.messages.create(
        model="claude-sonnet-4-5", max_tokens=300, tools=tools, messages=messages
    )
    print(final.content[0].text)
else:
    print(response.content[0].text)