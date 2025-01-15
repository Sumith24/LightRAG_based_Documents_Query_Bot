import logging
import numpy as np
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import hf_model_complete, hf_embedding
from transformers import AutoModel, AutoTokenizer
from openai import AzureOpenAI
from constants import AZURE_OPENAI_API_VERSION,AZURE_OPENAI_DEPLOYMENT,AZURE_OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT,AZURE_EMBEDDING_DEPLOYMENT,HUGF_MODEL_NAME,HUGF_EMBEDDING_MODEL

logging.basicConfig(level=logging.INFO)

def lightrag_with_huggingface_model(storage_path):
    light_rag = LightRAG(
        working_dir=storage_path,
        llm_model_func=hf_model_complete,
        llm_model_name=HUGF_MODEL_NAME,
        embedding_func=EmbeddingFunc(
            embedding_dim=384,
            max_token_size=5000,
            func=lambda texts: hf_embedding(
                texts,
                tokenizer=AutoTokenizer.from_pretrained(HUGF_EMBEDDING_MODEL),
                embed_model=AutoModel.from_pretrained(HUGF_EMBEDDING_MODEL),
            ),
        ),
    )
    return light_rag

def lightrag_with_azureopenai_model(storage_path):
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

        input_tokens = chat_completion.usage.prompt_tokens  # Tokens used in the input (prompt)
        output_tokens = chat_completion.usage.completion_tokens  # Tokens used in the output (response)

        print(f"\nInput-Tokens:{input_tokens}\nOutput-Tokens:{output_tokens}")

        total_cost = await calculate_cost(input_tokens, output_tokens)
        print(f"\nTokens-Cost-Per-Call:{total_cost}")

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
    
    async def calculate_cost(input_tokens, output_tokens):
        # Calculate the openai cost per call
        price_per_1M_input_tokens = 0.165
        price_per_1M_output_tokens = 0.66
        cost = ((input_tokens / 1000000) * price_per_1M_input_tokens)+((output_tokens / 1000000) * price_per_1M_output_tokens)
        return cost

    light_rag = LightRAG(
        working_dir=storage_path,
        llm_model_func=llm_model_func,
        embedding_func=EmbeddingFunc(
            embedding_dim=3072,
            max_token_size=8192,
            func=embedding_func,
        ),
    )
    return light_rag

def query_index(rag, query_text, search_method="naive"):
    if search_method == "naive":
        return rag.query(query_text, param=QueryParam(mode="naive"))
    if search_method == "local":
        return rag.query(query_text, param=QueryParam(mode="local"))
    if search_method == "global":
        return rag.query(query_text, param=QueryParam(mode="global"))
    if search_method == "hybrid":
        return rag.query(query_text, param=QueryParam(mode="hybrid"))
    if search_method == "mix":
        return rag.query(query_text, param=QueryParam(mode="mix"))
