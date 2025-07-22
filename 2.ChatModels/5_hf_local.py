from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task ="text-generation",
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)

model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of France?")
print(result.content)
# from transformers import pipeline
# from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

# # Build the HF Transformers pipeline manually
# pipe = pipeline(
#     task="text-generation",
#     model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     temperature=0.5,
#     max_new_tokens=100
# )

# # Wrap in LangChain
# llm = HuggingFacePipeline(pipeline=pipe)
# model = ChatHuggingFace(llm=llm)
