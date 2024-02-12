from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from .userProfile import user_profile
#from articles import article

load_dotenv()

# create a template
summ_template = """
    Summarize the key points of this article in bullet point format.
    Focusing on aspects most relevant the specified audience profile.
    Ensure the summary captures the essence of the article and its implications for this audience:
    {article}
    ---
    Audience Profile:
    {user_profile}
    ---
"""

# assign template to prompt
summ_prompt = ChatPromptTemplate.from_template(summ_template)

# declare a model
summ_model = (
    ChatOpenAI(temperature=0)
)

# declare an output parser
summ_output_parser = StrOutputParser()


def get_summary(article, user_profile):
    # setup chain
    summ_chain = (
        {"article": RunnablePassthrough(), "user_profile": RunnablePassthrough()}
        | summ_prompt
        | summ_model
        | summ_output_parser
    )
    
    # invoke chain
    summary = summ_chain.invoke(
        {
            "article": article,
            "user_profile": user_profile
        }
    )
    
    return summary