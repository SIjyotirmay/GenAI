from langchain_community.document_loaders import DirectoryLoader,PyPDFLoader
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
    template = 'Write a summary for the following pdf - \n {pdf}',
    input_variables=['pdf']
)

parser = StrOutputParser()


loader=DirectoryLoader(
    path = 'D:/GenAI-Journey/GenAI/pdf_storage',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()
#docs = loader.lazy_load() in for loop for multiple(100s or 1000s) pdfs 
print(len(docs))
print(docs[101].page_content)
print(docs[101].metadata)

chain = prompt | model1 | parser

#print('Answer: \n'+chain.invoke({'pdf':docs[1].page_content}))