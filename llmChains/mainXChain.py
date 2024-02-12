from dotenv import load_dotenv
from langchain.chains import SequentialChain
from .summaryChain import get_summary
from .ideaGenChain import get_tweet_ideas
from .tweetWriterChain import get_tweet_drafts
from .userProfile import user_profile
#from articles import article

load_dotenv()

def article2Tweet(article):
    summary = get_summary(article, user_profile)
    tweetIdeas = get_tweet_ideas(summary, user_profile)
    tweetDrafts = get_tweet_drafts(tweetIdeas, user_profile) 
    return tweetDrafts

def main(article):
    response = article2Tweet(article)
    return response

if __name__ == "__main__":
    main()