import os
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
import numpy as np
import logging
from openai import AzureOpenAI
from constants import AZURE_OPENAI_API_VERSION, AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_EMBEDDING_DEPLOYMENT

logging.basicConfig(level=logging.INFO)


WORKING_DIR = "./storage_1"

if os.path.exists(WORKING_DIR):
    import shutil

    shutil.rmtree(WORKING_DIR)

os.mkdir(WORKING_DIR)


async def llm_model_func(prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs) -> str:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    if history_messages:
        messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    chat_completion = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,  # model = "deployment_name".
        messages=messages,
        temperature=kwargs.get("temperature", 0),
        top_p=kwargs.get("top_p", 1),
        n=kwargs.get("n", 1),
    )
    return chat_completion.choices[0].message.content


async def embedding_func(texts: list[str]) -> np.ndarray:
    client = AzureOpenAI(
        api_key=AZURE_OPENAI_API_KEY,
        api_version=AZURE_OPENAI_API_VERSION,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
    )
    embedding = client.embeddings.create(model=AZURE_EMBEDDING_DEPLOYMENT, input=texts)

    embeddings = [item.embedding for item in embedding.data]
    return np.array(embeddings)


async def test_funcs():
    result = await llm_model_func("How are you?")
    print("Result of llm_model_func: ", result)

    result = await embedding_func(["How are you?"])
    print("Result of embedding_func: ", result.shape)
    print("Dimension of embedding: ", result.shape[1])


asyncio.run(test_funcs())

embedding_dimension = 3072

rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=llm_model_func,
    embedding_func=EmbeddingFunc(
        embedding_dim=embedding_dimension,
        max_token_size=8192,
        func=embedding_func,
    ),
)

book1 = open("./data/contract_sample_1.txt", encoding="utf-8")

rag.insert([book1.read()])

query_text = "What are the names of lessee and lessor?"

print("Result (Naive):")
print(rag.query(query_text, param=QueryParam(mode="naive")))

print("\nResult (Local):")
print(rag.query(query_text, param=QueryParam(mode="local")))

print("\nResult (Global):")
print(rag.query(query_text, param=QueryParam(mode="global")))

print("\nResult (Hybrid):")
print(rag.query(query_text, param=QueryParam(mode="hybrid")))