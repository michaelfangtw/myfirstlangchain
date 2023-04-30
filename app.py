import os
#change to env variable
#from apikey import apikey

import streamlit as st
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain,SequentialChain

# promt template
title_template =PromptTemplate(
    input_variables=['topic'],
    template ='write me a  english youtube video title about {topic} '
)

script_template =PromptTemplate(
    input_variables=['title'],
    template ='write me a english youtube video script base on this TITLE: {title}'
)

apikey=os.environ.get("OPENAI_API_KEY")
#print(apikey)


st.title("youtube gpt creater")
prompt=st.text_input("plug your prompt here")

# llm
llm= OpenAI(temperature=0.9)  
title_chain =LLMChain(llm=llm,prompt=title_template,verbose=True,output_key='title')
script_chain =LLMChain(llm=llm,prompt=script_template,verbose=True,output_key='script')
sequential_chain=SequentialChain(chains=[title_chain,script_chain],input_variables=['topic'],output_variables=['title','script'], verbose=True)


# show response
if prompt:
    response =sequential_chain({'topic':prompt})
    st.write(response['title'])
    print(response['title'])
    st.write(response['script'])
    print(response['script'])

