import os
import logging
import numpy as np
from lightrag import LightRAG
from lightrag.utils import EmbeddingFunc
from openai import AzureOpenAI
from constants import AZURE_OPENAI_API_VERSION,AZURE_OPENAI_DEPLOYMENT,AZURE_OPENAI_API_KEY,AZURE_OPENAI_ENDPOINT,AZURE_EMBEDDING_DEPLOYMENT
from bs4 import BeautifulSoup

logging.basicConfig(level=logging.INFO)

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
        print(f"\nAzure-call-cost:{total_cost}")

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



def main():
    import pdb;pdb.set_trace();
    # index_path = "contract_sample_2"
    # base_input_dir = './data'
    # if f'{index_path}.txt' in os.listdir(base_input_dir):
    #     rag = lightrag_with_azureopenai_model(index_path)
    #     with open(os.path.join(base_input_dir, f'{index_path}.txt'), "r", encoding="utf-8") as doc_f:
    #         rag.insert(doc_f.read())

    new_index = "new_index"
    rag = lightrag_with_azureopenai_model(new_index)
    for file in os.listdir('dummy_data'):
        with open(os.path.join(os.getcwd(), f"dummy_data\\{file}"), "r", encoding="utf-8") as doc_f:
            text = '\n'.join(BeautifulSoup(doc_f.read(), "html.parser").findAll(string=True))
            rag.insert(text)


if __name__=="__main__":
    main()