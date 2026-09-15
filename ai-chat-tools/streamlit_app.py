import os
import ast
import operator
import streamlit as st
import anthropic
from datetime import datetime
from dotenv import load_dotenv

# ---------- SETUP ----------
load_dotenv()
try:
    api_key = st.secrets["ANTHROPIC_API_KEY"]   # Streamlit Cloud
except Exception:
    api_key = os.getenv("ANTHROPIC_API_KEY")    # local .env fallback

client = anthropic.Anthropic(api_key=api_key)
MODEL = "claude-sonnet-5"

# ---------- TOOLS ----------
def get_current_time():
    return datetime.now().strftime("%A, %B %d %Y, %I:%M %p")

def calculate(expression):
    ops = {ast.Add: operator.add, ast.Sub: operator.sub,
           ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.Pow: operator.pow, ast.Mod: operator.mod,
           ast.USub: operator.neg}
    def _eval(node):
        if isinstance(node, ast.Constant):        # numbers
            return node.value
        if isinstance(node, ast.BinOp):
            return ops[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp):
            return ops[type(node.op)](_eval(node.operand))
        raise ValueError("unsupported expression")
    try:
        return str(_eval(ast.parse(expression, mode="eval").body))
    except Exception as e:
        return f"Error: {e}"

TOOLS = [
    {"name": "get_current_time",
     "description": "Returns the current date and time.",
     "input_schema": {"type": "object", "properties": {}, "required": []}},
    {"name": "calculate",
     "description": "Evaluates a math expression, e.g. '23*7'.",
     "input_schema": {"type": "object",
                      "properties": {"expression": {"type": "string"}},
                      "required": ["expression"]}},
]

def run_tool(name, tool_input):
    if name == "get_current_time":
        return get_current_time()
    if name == "calculate":
        return calculate(tool_input["expression"])
    return "Unknown tool"

# ---------- UI ----------
st.title("🤖 Kevin's AI Assistant")

# API history (raw blocks) — what we SEND to Claude
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display history (plain text) — what we SHOW on screen
if "display" not in st.session_state:
    st.session_state.display = []

for m in st.session_state.display:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])

if user_input := st.chat_input("Ask me anything..."):
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.display.append({"role": "user", "text": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    while True:
        response = client.messages.create(
            model=MODEL, max_tokens=1000,
            messages=st.session_state.messages, tools=TOOLS,
        )
        st.session_state.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            text = "".join(b.text for b in response.content if b.type == "text")
            st.session_state.display.append({"role": "assistant", "text": text})
            with st.chat_message("assistant"):
                st.markdown(text)
            break

        tool_results = []
        for b in response.content:
            if b.type == "tool_use":
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": b.id,
                    "content": str(run_tool(b.name, b.input)),
                })
        st.session_state.messages.append({"role": "user", "content": tool_results})