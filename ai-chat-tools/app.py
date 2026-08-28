import anthropic
from datetime import datetime
from dotenv import load_dotenv
 
load_dotenv()
# Create ONE client object we reuse for the whole program.
client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from your environment
MODEL = "claude-sonnet-5"

# ---------- 1. THE TWO TOOLS (plain Python functions) ----------
def get_current_time():
    now = datetime.now()
    return now.strftime("%A, %B %d %Y, %I:%M %p")

def calculate(expression):
    # NOTE: eval() is fine for learning but NOT safe for untrusted input.
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error: {e}"

# ---------- 2. TELL CLAUDE WHAT TOOLS EXIST (schemas) ----------
TOOLS = [
    {
        "name": "get_current_time",
        "description": "Returns the current date and time.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "calculate",
        "description": "Evaluates a math expression, e.g. '23*7'.",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string"}},
            "required": ["expression"],
        },
    },
]

# ---------- 3. MAP TOOL NAMES -> FUNCTIONS ----------
def run_tool(name, tool_input):
    if name == "get_current_time":
        return get_current_time()
    if name == "calculate":
        return calculate(tool_input["expression"])
    return "Unknown tool"

# ---------- 4. THE CHAT LOOP ----------
messages = []  # the whole conversation history lives here

print("Chat ready. Type 'quit' to exit.")
while True:
    user_input = input("\nYou: ")
    if user_input == "quit":
        break

    messages.append({"role": "user", "content": user_input})

    # Inner loop: keep going until Claude stops asking for tools.
    while True:
        response = client.messages.create(
            model=MODEL,
            max_tokens=1000,
            messages=messages,
            tools=TOOLS,
        )
        messages.append({"role": "assistant", "content": response.content})

        # If Claude gave a normal answer (no tool needed), print it and stop.
        if response.stop_reason != "tool_use":
            for block in response.content:
                if block.type == "text":
                    print(f"\nClaude: {block.text}")
            break

        # Otherwise Claude asked for a tool -> run each one, collect results.
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = run_tool(block.name, block.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": str(result),
                })

        # Send the tool results back so Claude can finish its answer.
        messages.append({"role": "user", "content": tool_results})