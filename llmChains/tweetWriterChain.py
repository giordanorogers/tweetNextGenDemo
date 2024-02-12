from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .userProfile import user_profile
# from .ideaGenChain import tweetIdeas

load_dotenv()

# create a template
tweet_template = """
    From the list of tweet ideas below, select the three most compelling ones for the specified audience. 
    Refine these tweets to enhance their clarity, impact, and engagement potential. 
    Each tweet should be below 280 characters and contain absolutely no hashtags or links.
    NO HASHTAGS!
    Ensure each tweet is polished and directly speaks to the interests and concerns of the audience:
    {tweetIdeas}
    ---
    Audience Profile:
    {user_profile}
    ---
    Output only your final revised tweets below.
    Don't include any other text in your response.
    
    THE THREE FINAL TWEETS:
"""

# assign template to prompt
tweet_prompt = ChatPromptTemplate.from_template(tweet_template)

# declare a model
tweet_model = (
    ChatOpenAI(temperature=0.8)
)

# declare an output parser
tweet_output_parser = StrOutputParser()


def get_tweet_drafts(tweetIdeas, user_profile):
    # setup chain
    tweet_chain = (
        {"tweetIdeas": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
        | tweet_prompt
        | tweet_model
        | tweet_output_parser
    )
    
    # invoke chain
    tweetDrafts = tweet_chain.invoke(
        {
            "tweetIdeas": tweetIdeas,
            "user_profile": user_profile
        }
    )
    
    return tweetDrafts