from langchain_groq import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
from config import Config
import json
def get_decision_maker():
    llm = ChatGroq(
        groq_api_key=Config.GROQ_API_KEY, 
        model_name=Config.MODEL_NAME,
        temperature=0.7
    )

    def decision_maker_node(state):
        system_prompt = """You are an intent classifier. Analyze the user's request and determine what they want.
        
        Return a JSON object with these boolean fields:
        - needs_content: true if they want original content created
        - needs_summary: true if they want a summary
        - needs_social_media: true if they want a social media post
        - needs_research: true if they want research/information gathering
        
        Examples:
        - "Create a blog post about AI" -> {"needs_content": true, "needs_summary": false, "needs_social_media": false, "needs_research": false}
        - "Summarize this for me: [content]" -> {"needs_content": false, "needs_summary": true, "needs_social_media": false, "needs_research": false}
        - "Create a Twitter post about climate change" -> {"needs_content": false, "needs_summary": false, "needs_social_media": true, "needs_research": false}
        - "Research the latest developments in quantum computing" -> {"needs_content": false, "needs_summary": false, "needs_social_media": false, "needs_research": true}
        - "Create content about space exploration and make a social post" -> {"needs_content": true, "needs_summary": false, "needs_social_media": true, "needs_research": false}
        
        Only return the JSON object, nothing else."""

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state.get("user_input", ""))
        ]

        response = llm.invoke(messages)

        try:
            decision = json.loads(response.content)
            state["decisions"] = decision
        except:
            
            state["decisions"] = {
                "needs_content": True,
                "needs_summary": True, 
                "needs_social_media": True,
                "needs_research": True
            }
        
        return state
    
    return decision_maker_node

