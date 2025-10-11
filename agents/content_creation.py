from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config


def get_content_creation_agent():
    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY, 
        model_name=Config.MODEL_NAME,
        temperature=0.7
        )
    def content_creation_node(state):
        system_prompt = """ You are a content creation agent. Your job is to create engaging, 
        informative content based on the user's input. Generate comprehensive content that 
        can be used for various purposes including summaries and social media posts."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get("user_input", ""))
        ]

        response = llm.invoke(messages)
        state["content"] = response.content
        return state
    return content_creation_node