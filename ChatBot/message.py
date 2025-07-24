from langchain_core.messages import SystemMessage,HumanMessage,AIMessage
from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
from dotenv import load_dotenv

load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task ="text-generation",
)


model = ChatHuggingFace(llm=llm)
 

messages = [
    SystemMessage(content="You are a knowledgeable AI assistant who answers truthfully and accurately."),
    HumanMessage(content=(
        "What is LangChain?\n"
        "Explain it as a Python-based framework used for building applications with large language models (LLMs). "
        "Mention key features like chains, agents, memory, tools, and retrieval-augmented generation (RAG)."
    ))
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)