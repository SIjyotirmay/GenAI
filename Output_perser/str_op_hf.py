from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

load_dotenv()

# Load the local pipeline-based model
llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

# Prompt templates
template1 = PromptTemplate(
    template='write a detailed report on {topic}',
    input_variables=['topic']
)

template2 = PromptTemplate(
    template='write a 5 line summary on the following text:\n{text}',
    input_variables=['text']
)

# Output parser
parser = StrOutputParser()

# Complete chain in one go
chain = (
    template1 
    | model 
    | parser 
    | RunnableLambda(lambda x: {"text": x}) 
    | template2 
    | model 
    | parser
)

# Run it
result = chain.invoke({'topic': 'black hole'})

print(result)

# What RunnableLambda does here:
# It takes the string output from the first parser and wraps it into a dict like this: