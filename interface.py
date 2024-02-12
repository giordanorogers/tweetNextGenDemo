import os
import streamlit as st
from dotenv import load_dotenv
from llmChains import mainXChain


# Load API Key from Environment Variable
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Set Page Title and Configuration
st.set_page_config(page_title="Tweet-Next-Gen Demo")
st.title('Tweet-Next-Gen Demo')

def generate_response(article):
  st.info(mainXChain.main(article))

with st.form('my_form'):
  article = st.text_area('Enter text:', 'Paste an article here.')
  submitted = st.form_submit_button('Submit')
  if not OPENAI_API_KEY.startswith('sk-'):
    st.warning('Please enter your OpenAI API key!', icon='⚠')
  if submitted and OPENAI_API_KEY.startswith('sk-'):
    generate_response(article)