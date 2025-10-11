from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

def get_research_assistant():
    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY, 
        model_name=Config.MODEL_NAME,
        temperature=0.7
        )
    def research_assistant_node(state):
        system_prompt = """ You are a research assistant. Your job is to research the user's input. """
        
        content = state.get("content", "")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"what research paper should we explore for {content}")
        ]

        response = llm.invoke(messages)
        state["research_topics"] = response.content
        return state
    return research_assistant_node
