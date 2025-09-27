from langchain_community.document_loaders import WebBaseLoader
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
    template = 'Answer the following question - \n {question} from the following text - \n {text}',
    input_variables=['question','text']
)

parser = StrOutputParser()
url = 'https://www.flipkart.com/asus-expertbook-p1-high-performance-processor-intel-core-i7-13th-gen-13620h-16-gb-512-gb-ssd-windows-11-home-p1503cva-s71042ws-thin-light-laptop/p/itmaaf79b8a24f17?pid=COMH8Z2QK37PNMFZ'
loader = WebBaseLoader(url)

docs = loader.load()

# print(len(docs))

# print(docs[0].page_content)

chain = prompt | model1 | parser

print(chain.invoke({'question':'What is the peak brightness of this product?','text':docs[0].page_content}))
