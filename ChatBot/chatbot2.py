from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from dotenv import load_dotenv
load_dotenv()

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task ="text-generation",
    pipeline_kwargs=dict(
        temperature=1.5,
        max_new_tokens=20
    )
)

model = ChatHuggingFace(llm=llm)


chat_history = [
    SystemMessage(content='You are a helpful AI assistant')
]
while True:
    user_input = input('You: ')
    chat_history.append(HumanMessage(content=user_input))
    if user_input == 'exit':
        break
    result = model.invoke(user_input)
    chat_history.append(AIMessage(content=result.content))
    print("AI: ",result.content)


print(chat_history)