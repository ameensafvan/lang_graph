from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages


from agents.content_creation import get_content_creation_agent
from agents.summerization import get_summerization_agent
from agents.social_media import get_social_media_agent
from agents.research_assistant import get_research_assistant
from tools.wiki import wiki_research_node
from tools.arxiv import arxiv_research_node

class State(TypedDict):
    user_input: str
    content: str
    summary: str
    social_post: str
    research_topics: str
    wiki_research: str
    arxiv_research: str
    research: str
    decisions: dict
    messages: Annotated[list, add_messages]


def combine_research(state):
    """Combine Wikipedia and Arxiv research results."""
    wiki_data = state.get("wiki_research", "")
    arxiv_data = state.get("arxiv_research", "")
    
    combined_research = f"""
    Wikipedia Research:
    {wiki_data}
    
    Arxiv Research:
    {arxiv_data}
    """
    
    state["research"] = combined_research
    return state

def after_decision(state):

    dec = state.get("decisions", {})

    next_node = []

    if dec.get("needs_content", False):
        next_node.append("content_creation")
    elif dec.get()