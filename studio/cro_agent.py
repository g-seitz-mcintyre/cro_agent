from dotenv import load_dotenv
from firecrawl import FirecrawlApp
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

load_dotenv()


class CROGraphState(TypedDict):
    url: str
    html: str
    markdown: str
    screenshot_url: str
    html_analysis: str
    markdown_analysis: str
    vision_analysis: str
    cro_summary: str
    cro_report: bytes


# Node: Scrape website using Firecrawl
def firecrawl_node(state):
    url = state["url"]
    app = FirecrawlApp()
    scrape_result = app.scrape_url(
        url, formats=["markdown", "html", "screenshot@fullPage"]
    )

    return {
        "html": scrape_result.html,
        "markdown": scrape_result.markdown,
        # Firecrawl returns screenshot URL under 'screenshotUrl'
        "screenshot_url": scrape_result.screenshot,
    }


# Node: Analyze HTML content for CRO
def html_agent_node(state):
    llm = ChatOpenAI(model="gpt-4")
    system_msg = SystemMessage(
        content=(
            "You are a senior conversion-rate-optimization (CRO) consultant "
            "with a strong background in technical SEO, web performance and UX. "
            "Your job is to review raw HTML and deliver concise, evidence-based "
            "recommendations that could measurably increase conversions. "
            "Prioritise the highest-impact fixes and keep each recommendation actionable."
        )
    )

    human_prompt = (
        "You will receive the raw HTML of a single web page delimited by ``<html_source>`` tags.\n\n"
        "**Tasks**:\n"
        "1. Identify up to **10 conversion-blocking or conversion-limiting issues**.\n"
        "2. For each issue provide the following as a markdown table with one row per issue: \n"
        "   • **Issue** (short title)\n"
        "   • **Severity** (1-5)\n"
        "   • **Why it hurts conversion** (1-2 sentences)\n"
        "   • **Actionable fix** (concrete next step)\n"
        "   • **Expected impact** (qualitative: low / medium / high).\n"
        "3. Finish with a three-sentence executive summary highlighting the **top 3 fixes** in order of potential conversion lift.\n\n"
        "<html_source>\n"
        f"{state['html'][:20000]}\n"
        "</html_source>"
    )

    response = llm([system_msg, HumanMessage(content=human_prompt)])
    return {"html_analysis": response.content}


# Node: Analyze Markdown content for CRO
def md_agent_node(state):
    llm = ChatOpenAI(model="gpt-4.1")
    system_msg = SystemMessage(
        content=(
            "You are a veteran CRO copywriter and messaging strategist. "
            "Evaluate tone, clarity, information hierarchy and call-to-action (CTA) effectiveness. "
            "Deliver crisp, high-impact copy recommendations that can boost user motivation and minimise friction."
        )
    )

    human_prompt = (
        "Review the Markdown content below, which represents the **visible copy** of the target web page.\n\n"
        "**Tasks**:\n"
        "1. Detect up to **10 copy or messaging issues** that may reduce conversions (e.g., vague CTA text, missing risk-reducers, weak value proposition).\n"
        "2. Provide a markdown table with columns: **Issue • Current text (snippet) • Recommended rewrite • Psychological principle leveraged • Expected impact**.\n"
        "3. Conclude with a short priority list sorted by **highest impact / lowest effort**.\n\n"
        "```markdown\n"
        f"{state['markdown'][:20000]}\n"
        "```"
    )

    response = llm([system_msg, HumanMessage(content=human_prompt)])
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
                    "You are a CRO art director analysing the **visual hierarchy, colour contrast, trust cues, and CTA prominence** of the screenshot.\n\n"
                    "List up to **8 visual issues** in a markdown table with columns: **Issue • Why it hurts conversion • Specific redesign suggestion • Expected impact**.\n"
                    "Finish with a one-paragraph summary of how these visual changes together could improve user journey and conversion rate."
                ),
            },
        ]
    )

    response = llm.invoke([vision_prompt])
    return {"vision_analysis": response.content}


# Node: Aggregate all analyses
def aggregate_node(state: CROGraphState) -> dict:
    """Combine all individual analyses into a single markdown summary."""

    llm = ChatOpenAI(model="o4-mini")

    combined = "\n\n".join(
        [
            "## HTML (Technical & UX) Analysis\n" + state["html_analysis"],
            "## Copy & Messaging Analysis\n" + state["markdown_analysis"],
            "## Visual Hierarchy Analysis\n" + state["vision_analysis"],
        ]
    )

    prompt = HumanMessage(
        content=(
            "Create a final CRO Audit report based on these analyses provided by your analysts:\n\n"
            f"{combined}"
        )
    )

    system = SystemMessage(
        content=(
            "You are a seasoned technical writer with lots of experience in creating actionable CRO audit reports."
        )
    )

    response = llm([system, prompt])
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
