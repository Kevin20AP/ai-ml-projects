from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()


# ── 1. STATE DEFINITION ───────────────────────────────────────────────────────
class SupportState(TypedDict):
    messages: List
    intent: str
    confidence: float
    response: str
    escalate: bool


# ── 2. LLM SETUP ──────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)


# ── 3. NODE: INTENT CLASSIFIER ────────────────────────────────────────────────
def classify_intent(state: SupportState) -> SupportState:
    user_message = state["messages"][-1].content

    prompt = f"""Classify this customer message into exactly one category:
- billing: payment, invoice, charge, refund, subscription
- technical: bug, error, not working, crash, login issue  
- general: question, information, how to, feature request
- unknown: cannot determine

Message: "{user_message}"

Reply with ONLY the category word and a confidence score 0-1.
Format: category|confidence
Example: billing|0.92"""

    response = llm.invoke([HumanMessage(content=prompt)])

    try:
        parts = response.content.strip().split("|")
        intent = parts[0].strip().lower()
        confidence = float(parts[1].strip())
    except:
        intent = "unknown"
        confidence = 0.0

    print(f"\n🔍 Intent classified: {intent} (confidence: {confidence})")
    return {**state, "intent": intent, "confidence": confidence}


# ── 4. NODE: BILLING HANDLER ──────────────────────────────────────────────────
def handle_billing(state: SupportState) -> SupportState:
    user_message = state["messages"][-1].content

    prompt = f"""You are a billing support agent. Help with this billing issue.
Be helpful, professional, and specific.

Customer message: {user_message}

Provide a clear response about billing, payments, or refunds."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("💳 Billing handler responding...")
    return {**state, "response": response.content, "escalate": False}


# ── 5. NODE: TECHNICAL HANDLER ────────────────────────────────────────────────
def handle_technical(state: SupportState) -> SupportState:
    user_message = state["messages"][-1].content

    prompt = f"""You are a technical support agent. Help troubleshoot this issue.
Provide step-by-step solutions where possible.

Customer message: {user_message}

Provide clear technical assistance."""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("🔧 Technical handler responding...")
    return {**state, "response": response.content, "escalate": False}


# ── 6. NODE: GENERAL HANDLER ──────────────────────────────────────────────────
def handle_general(state: SupportState) -> SupportState:
    user_message = state["messages"][-1].content

    prompt = f"""You are a helpful customer support agent.
Answer this general question clearly and helpfully.

Customer message: {user_message}"""

    response = llm.invoke([HumanMessage(content=prompt)])
    print("💬 General handler responding...")
    return {**state, "response": response.content, "escalate": False}


# ── 7. NODE: HUMAN ESCALATION ─────────────────────────────────────────────────
def escalate_to_human(state: SupportState) -> SupportState:
    print("🚨 Escalating to human agent...")
    response = (
            "I'm sorry, I wasn't able to confidently understand your request. "
            "I'm escalating you to a human agent who will assist you shortly. "
            "Your ticket ID is #TKT-" + str(abs(hash(state["messages"][-1].content)) % 10000) +
            ". Expected wait time: 2-3 minutes."
    )
    return {**state, "response": response, "escalate": True}


# ── 8. ROUTING LOGIC ──────────────────────────────────────────────────────────
def route_by_intent(state: SupportState) -> str:
    intent = state.get("intent", "unknown")
    confidence = state.get("confidence", 0.0)

    # Low confidence = escalate to human
    if confidence < 0.6 or intent == "unknown":
        print(f"⚠️  Low confidence ({confidence}) — routing to human")
        return "escalate"

    print(f"✅ Routing to: {intent}")
    return intent


# ── 9. BUILD THE GRAPH ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(SupportState)

    # Add nodes
    graph.add_node("classify", classify_intent)
    graph.add_node("billing", handle_billing)
    graph.add_node("technical", handle_technical)
    graph.add_node("general", handle_general)
    graph.add_node("escalate", escalate_to_human)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        route_by_intent,
        {
            "billing": "billing",
            "technical": "technical",
            "general": "general",
            "escalate": "escalate"
        }
    )

    # All handlers end the conversation
    graph.add_edge("billing", END)
    graph.add_edge("technical", END)
    graph.add_edge("general", END)
    graph.add_edge("escalate", END)

    return graph.compile()


# ── 10. MAIN CHAT LOOP ────────────────────────────────────────────────────────
app = build_graph()

print("\n" + "=" * 50)
print("🤖 LangGraph Support Bot Ready! Type 'quit' to exit.")
print("=" * 50)
print("\nTry asking:")
print("  - 'I was charged twice this month'")
print("  - 'My app keeps crashing on login'")
print("  - 'How do I reset my password?'")
print("  - 'asdfghjkl' (tests escalation)\n")

while True:
    user_input = input("You: ").strip()
    if user_input.lower() == "quit":
        print("Goodbye!")
        break
    if not user_input:
        continue

    initial_state = SupportState(
        messages=[HumanMessage(content=user_input)],
        intent="",
        confidence=0.0,
        response="",
        escalate=False
    )

    result = app.invoke(initial_state)
    print(f"\n🤖 Bot: {result['response']}\n")
    print("-" * 50)
