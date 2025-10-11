from langchain_community.tools import ArxivQueryRun
from langchain.utilities import ArxivAPIWrapper

def get_arxiv_tool():
    arxiv_api = ArxivAPIWrapper()
    return ArxivQueryRun(api_wrapper=arxiv_api)

def arxiv_research_node(state):
    tool = get_arxiv_tool()
    res_topics = state.get("research_topics", "")

    last_topic = res_topics.split()[-1].strip()

    if last_topic:
        try:
            result = tool.run(last_topic)
            state["arxiv_research"] = f"Topic: {last_topic}\nResult:{result}"
        except Exception as e:
            state["arxiv_research"] = f"Error: {str(e)}"
    else:
        state["arxiv_research"] = "No more topics to research"
    return state