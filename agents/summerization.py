from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from config import Config

def get_summerization_agent():
    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY, 
        model_name=Config.MODEL_NAME,
        temperature=0.7
        )
    def summerization_node(state):
        system_prompt = """ You are a summerization agent. Your job is to summarize the user's input. """

        content_to_summerize = state.get("content", "")
        research_data = state.get("research_data", "")

        prompt = f"""
        Content to summerize: {content_to_summerize}
        
        Research data: {research_data}
        
        please create a comprehensive summary.
        """
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=prompt)
        ]

        response = llm.invoke(messages)
        state["summary"] = response.content
        return state
    return summerization_node
