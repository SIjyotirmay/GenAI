# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import SystemMessage,HumanMessage

# chat_template = ChatPromptTemplate([
#     SystemMessage(content='You are a helpful {domain} expert'),
#     HumanMessage(content='Explain in simple terms, What is {topic}')
# ])

# prompt =  chat_template.invoke({'domain':'cricket','topic':'Dusra'})

# print(prompt)

from langchain_core.prompts import ChatPromptTemplate
 
chat_template = ChatPromptTemplate([
     ('system','you are a helpful {domain} expert'),
     ('human','Explain in simple terms, what is {topic}')

])

prompt =  chat_template.invoke({'domain':'cricket','topic':'Dusra'})

print(prompt)