from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage
from langgraph.prebuilt import create_react_agent
import requests
import wikipedia

load_dotenv()

# ── 1. TOOLS ──────────────────────────────────────────────────────────────────

@tool
def calculator(expression: str) -> str:
    """Evaluates a math expression. Input should be a math expression like 20 * 0.15"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


@tool
def weather(city: str) -> str:
    """Gets current weather for a city. Input should be a city name like Dallas"""
    try:
        url = f"https://wttr.in/{city}?format=3"
        response = requests.get(url, timeout=5)
        return response.text
    except Exception as e:
        return f"Could not fetch weather: {str(e)}"


@tool
def wikipedia_search(query: str) -> str:
    """Looks up facts about people, places, or concepts."""
    try:
        result = wikipedia.summary(query, sentences=3)
        return result
    except wikipedia.DisambiguationError as e:
        return wikipedia.summary(e.options[0], sentences=3)
    except Exception as e:
        return f"Could not find information: {str(e)}"


# ── 2. LLM + TOOLS ────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [calculator, weather, wikipedia_search]

# ── 3. CREATE REACT AGENT (LangGraph style) ───────────────────────────────────
agent = create_react_agent(llm, tools)

# ── 4. RUN FUNCTION ───────────────────────────────────────────────────────────
def run_agent(user_input: str):
    print(f"\nQuestion: {user_input}")
    print("-" * 55)

    result = agent.invoke({
        "messages": [HumanMessage(content=user_input)]
    })

    # Print the thought process
    for message in result["messages"]:
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tc in message.tool_calls:
                print(f"\nThought: I need to use {tc['name']}")
                print(f"Action: {tc['name']}")
                print(f"Action Input: {tc['args']}")
        elif hasattr(message, "name") and message.name:
            print(f"Observation: {message.content}")

    # Final answer is always the last message
    final = result["messages"][-1].content
    print(f"\nFinal Answer: {final}")
    print("=" * 55)


# ── 5. CHAT LOOP ──────────────────────────────────────────────────────────────
print("\n" + "="*55)
print("ReAct Agent Ready! Type 'quit' to exit.")
print("="*55)
print("\nTry asking:")
print("  - 'What is the weather in Dallas?'")
print("  - 'What is 15% of 340?'")
print("  - 'Who is Elon Musk?'")
print("  - 'What is the weather in Tokyo and what is 25% of 200?'")
print()

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    if not user_input:
        continue

    try:
        run_agent(user_input)
    except Exception as e:
        print(f"Error: {str(e)}\n")
