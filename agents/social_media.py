from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

def get_social_media_agent():
    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY, 
        model_name=Config.MODEL_NAME,
        temperature=0.7
        )
    def social_media_node(state):
        system_prompt = """ You are a social media expert. Create exciting posts from the content. I provide 
        that people will want to share. Make them catchy, include hashtags, and get more engagement."""

        content = state.get("content", "")

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=content)
        ]

        response = llm.invoke(messages)
        state["social_post"] = response.content
        return state
    return social_media_node