from langchain_community.document_loaders import PyPDFLoader
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


loader=PyPDFLoader('D:/GenAI-Journey/GenAI/pdf_storage/GradCam.pdf')
docs = loader.load()

#print(docs[0].page_content)
print(docs[1].metadata)

chain = prompt | model1 | parser


print('Answer: \n'+chain.invoke({'pdf':docs[1].page_content}))