from dotenv import load_dotenv
from firecrawl import Firecrawl
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

try:
    from dotenv import load_dotenv

    load_dotenv()  # pulls variables from .env during local dev
except ModuleNotFoundError:
    # running in Cloud where python-dotenv isn’t installed
    pass


class CROGraphState(TypedDict):
    url: str
    html: str
    markdown: str
    screenshot_url: str
    html_analysis: str
    markdown_analysis: str
    vision_analysis: str
    cro_summary: str


# Node: Scrape website using Firecrawl
def firecrawl_node(state):
    url = state["url"]
    app = Firecrawl()
    scrape_result = app.scrape(
        url,
        formats=["markdown", "html", {"type": "screenshot", "fullPage": True}],
        only_main_content=True,
    )

    return {
        "html": scrape_result.html,
        "markdown": scrape_result.markdown,
        # Firecrawl returns screenshot URL under 'screenshotUrl'
        "screenshot_url": scrape_result.screenshot,
    }


# Node: Analyze HTML content for CRO
def html_agent_node(state):
    llm = ChatOpenAI(model="gpt-4.1")
    system_msg = SystemMessage(
        content=(
            "You are a senior conversion-rate-optimization (CRO) consultant "
            "with a strong background in technical SEO, web performance and UX. "
            "Your job is to review raw HTML and deliver concise, evidence-based "
            "recommendations that could measurably increase conversions. "
            "Prioritise the highest-impact fixes and keep each recommendation actionable."
        )
    )

    human_prompt = f"{state['html']}\n"

    response = llm.invoke([system_msg, HumanMessage(content=human_prompt)])
    return {"html_analysis": response.content}


# Node: Analyze Markdown content for CRO
def md_agent_node(state):
    llm = ChatOpenAI(model="gpt-4.1")
    system_msg = SystemMessage(
        content=(
            "You are a veteran CRO copywriter and messaging strategist. "
            "Evaluate tone, clarity, information hierarchy and call-to-action (CTA) effectiveness on this website. "
            "Deliver crisp, high-impact copy recommendations that can boost user motivation and engagement."
        )
    )

    human_prompt = f"{state['markdown']}"

    response = llm.invoke([system_msg, HumanMessage(content=human_prompt)])
    return {"markdown_analysis": response.content}


# Node: Analyze screenshot for visual CRO cues
def vision_agent_node(state):
    llm = ChatOpenAI(model="gpt-4o")
    vision_prompt = HumanMessage(
        content=[
            {
                "type": "image_url",
                "image_url": {"url": state["screenshot_url"]},
            },
            {
                "type": "text",
                "text": (
                    "You are a CRO specialist analysing the **visual hierarchy, colour contrast, trust cues, and CTA prominence** of the screenshot.\n\n"
                    "List up to **8 visual issues** in a markdown table with columns: **Issue • Why it hurts conversion • Specific redesign suggestion • Expected impact**."
                ),
            },
        ]
    )

    response = llm.invoke([vision_prompt])
    return {"vision_analysis": response.content}


# Node: Aggregate all analyses
def aggregate_node(state: CROGraphState) -> dict:
    """Combine all individual analyses into a single markdown summary."""

    llm = ChatOpenAI(model="gpt-4o")

    combined = "\n\n".join(
        [
            "## HTML (Technical & UX) Analysis\n" + state["html_analysis"],
            "## Copy & Messaging Analysis\n" + state["markdown_analysis"],
            "## Visual Hierarchy Analysis\n" + state["vision_analysis"],
        ]
    )

    prompt = HumanMessage(
        content=(
            "Synthesize these analyses into a concise, actionable CRO gameplan:\n\n"
            f"{combined}"
        )
    )

    system = SystemMessage(
        content=(
            "You are a seasoned technical writer with lots of experience in formulating actionable instructions for conversion rate optimization."
        )
    )

    response = llm.invoke([system, prompt])
    return {"cro_summary": response.content}


# Build and compile the LangGraph
def build_graph():
    builder = StateGraph(CROGraphState)
    builder.set_entry_point("firecrawl")

    # Register nodes
    builder.add_node("firecrawl", firecrawl_node)
    builder.add_node("html_agent", html_agent_node)
    builder.add_node("md_agent", md_agent_node)
    builder.add_node("vision_agent", vision_agent_node)
    builder.add_node("aggregate", aggregate_node)

    # Define edges
    builder.add_edge(START, "firecrawl")
    builder.add_edge("firecrawl", "html_agent")
    builder.add_edge("firecrawl", "md_agent")
    builder.add_edge("firecrawl", "vision_agent")
    builder.add_edge("html_agent", "aggregate")
    builder.add_edge("md_agent", "aggregate")
    builder.add_edge("vision_agent", "aggregate")
    builder.add_edge("aggregate", END)

    return builder.compile()


graph = build_graph()
