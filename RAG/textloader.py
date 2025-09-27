from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()


llm1= HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)

model1 = ChatHuggingFace(llm = llm1)

prompt = PromptTemplate(
    template = 'Write a summary for the following poem - \n {poem}',
    input_variables=['poem']
)

parser = StrOutputParser()


loader=TextLoader('Cricket.txt',encoding = 'utf-8')
docs = loader.load()
#print(docs)
print(type(docs))
#print(len(docs))
#print(docs[0])
print(type(docs[0]))

print(docs[0].page_content)
print(docs[0].metadata)

chain = prompt | model1 | parser


print('Answer: \n'+chain.invoke({'poem':docs[0].page_content}))