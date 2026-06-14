from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
from typing import TypedDict, List

load_dotenv()

# ── 1. STATE ──────────────────────────────────────────────────────────────────
class ResearchState(TypedDict):
    topic: str
    research: str
    report: str
    feedback: str
    revision_count: int
    final_report: str

# ── 2. LLM ────────────────────────────────────────────────────────────────────
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)

# ── 3. AGENT 1 — RESEARCHER ───────────────────────────────────────────────────
def researcher_agent(state: ResearchState) -> ResearchState:
    print("\n🔍 Researcher Agent working...")

    messages = [
        SystemMessage(content="""You are an expert researcher. 
Your job is to gather comprehensive information on a topic.
Provide detailed facts, statistics, key concepts, and important points.
Structure your research clearly with sections."""),
        HumanMessage(content=f"Research this topic thoroughly: {state['topic']}")
    ]

    response = llm.invoke(messages)
    print("✅ Research complete")
    return {**state, "research": response.content}

# ── 4. AGENT 2 — WRITER ───────────────────────────────────────────────────────
def writer_agent(state: ResearchState) -> ResearchState:
    print("\n✍️  Writer Agent working...")

    # Include feedback if this is a revision
    feedback_section = ""
    if state.get("feedback"):
        feedback_section = f"\n\nPrevious feedback to address:\n{state['feedback']}"

    messages = [
        SystemMessage(content="""You are an expert technical writer.
Your job is to transform research into a clear, well-structured report.
Use headers, bullet points, and clear language.
Make it professional and easy to read."""),
        HumanMessage(content=f"""Write a professional report based on this research:

{state['research']}
{feedback_section}

Format the report with:
- Executive Summary
- Key Findings
- Detailed Analysis
- Conclusion""")
    ]

    response = llm.invoke(messages)
    print("✅ Report written")
    return {**state, "report": response.content}

# ── 5. AGENT 3 — CRITIC ───────────────────────────────────────────────────────
def critic_agent(state: ResearchState) -> ResearchState:
    print("\n🔎 Critic Agent reviewing...")

    messages = [
        SystemMessage(content="""You are a strict quality control editor.
Your job is to review reports and identify gaps, errors, or improvements.
Be specific about what needs to be fixed.
If the report is excellent, respond with exactly: APPROVED
If it needs work, list specific issues starting with: NEEDS REVISION"""),
        HumanMessage(content=f"""Review this report about {state['topic']}:

{state['report']}

Check for:
- Completeness and accuracy
- Clear structure and flow
- Missing important information
- Quality of conclusions""")
    ]

    response = llm.invoke(messages)
    feedback = response.content
    print(f"✅ Review complete: {'APPROVED ✅' if 'APPROVED' in feedback else 'NEEDS REVISION 🔄'}")
    return {**state, "feedback": feedback}

# ── 6. FINALIZE ───────────────────────────────────────────────────────────────
def finalize(state: ResearchState) -> ResearchState:
    print("\n📄 Finalizing report...")
    return {**state, "final_report": state["report"]}

# ── 7. ROUTING LOGIC ──────────────────────────────────────────────────────────
def should_revise(state: ResearchState) -> str:
    feedback = state.get("feedback", "")
    revision_count = state.get("revision_count", 0)

    # Max 2 revisions to prevent infinite loop
    if revision_count >= 2:
        print("\n⚠️  Max revisions reached — finalizing")
        return "finalize"

    if "APPROVED" in feedback:
        print("\n✅ Report approved — finalizing")
        return "finalize"
    else:
        print(f"\n🔄 Revision {revision_count + 1} needed")
        return "revise"

# ── 8. REVISION ROUTER ────────────────────────────────────────────────────────
def increment_revision(state: ResearchState) -> ResearchState:
    return {**state, "revision_count": state.get("revision_count", 0) + 1}

# ── 9. BUILD THE GRAPH ────────────────────────────────────────────────────────
def build_graph():
    graph = StateGraph(ResearchState)

    # Add all nodes
    graph.add_node("researcher", researcher_agent)
    graph.add_node("writer", writer_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("revise", increment_revision)
    graph.add_node("finalize", finalize)

    # Entry point
    graph.set_entry_point("researcher")

    # Fixed edges
    graph.add_edge("researcher", "writer")
    graph.add_edge("writer", "critic")
    graph.add_edge("revise", "writer")
    graph.add_edge("finalize", END)

    # Conditional edge after critic
    graph.add_conditional_edges(
        "critic",
        should_revise,
        {
            "finalize": "finalize",
            "revise":   "revise"
        }
    )

    return graph.compile()

# ── 10. MAIN ──────────────────────────────────────────────────────────────────
app = build_graph()

print("\n" + "="*55)
print("🤖 Multi-Agent Research System Ready!")
print("="*55)
print("\nTry topics like:")
print("  - 'Impact of RAG on enterprise AI'")
print("  - 'How do AI agents work'")
print("  - 'Future of large language models'")
print()

while True:
    topic = input("Research topic: ").strip()
    if topic.lower() == "quit":
        print("Goodbye!")
        break
    if not topic:
        continue

    print(f"\n{'='*55}")
    print(f"Starting research on: {topic}")
    print(f"{'='*55}")

    initial_state = ResearchState(
        topic=topic,
        research="",
        report="",
        feedback="",
        revision_count=0,
        final_report=""
    )

    try:
        result = app.invoke(initial_state)
        print(f"\n{'='*55}")
        print("FINAL REPORT:")
        print(f"{'='*55}")
        print(result["final_report"])
        print(f"\nRevisions made: {result['revision_count']}")
        print(f"{'='*55}\n")
    except Exception as e:
        print(f"Error: {str(e)}\n")
