from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .userProfile import user_profile
# from .summaryChain import summary

load_dotenv()

# create a template
idea_template = """
    Based on the summary below, generate 10 tweet ideas that resonate with the specified user profile. 
    Each idea should be concise, under 280 characters, and tailored to engage this audience effectively. 
    Highlight the core message or insight from the article summary:
    {summary}
    ---
    Audience Profile:
    {user_profile}
    ---
"""

# assign template to prompt
idea_prompt = ChatPromptTemplate.from_template(idea_template)

# declare a model
idea_model = (
    ChatOpenAI(temperature=0.5)
)

# declare an output parser
idea_output_parser = StrOutputParser()


def get_tweet_ideas(summary, user_profile):
    # setup chain
    idea_chain = (
        {"summary": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
        | idea_prompt
        | idea_model
        | idea_output_parser
    )
    
    # invoke chain
    tweetIdeas = idea_chain.invoke(
        {
            "summary": summary,
            "user_profile": user_profile
        }
    )
    
    return tweetIdeas