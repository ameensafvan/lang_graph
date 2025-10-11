from langchain_community.tools import WikipediaQueryRun
from langchain.utilities import WikipediaAPIWrapper

def get_wiki_tool():
    wikipedia_api = WikipediaAPIWrapper()
    return WikipediaQueryRun(api_wrapper=wikipedia_api)

def wiki_research_node(state):
    tool = get_wiki_tool()
    res_topics = state.get("research_topics", "")

    last_topic = res_topics.split()[-1].strip()

    if last_topic:
        try:
            result = tool.run(last_topic)
            state["wiki_research"] = f"Topic: {last_topic}\nResult:{result}"
        except Exception as e:
            state["wiki_research"] = f"Error: {str(e)}"
    else:
        state["wiki_research"] = "No more topics to research"
    return state
