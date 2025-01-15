import streamlit as st
import os
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
            max_tokens=1000,
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

def query_index(rag, query_text, search_method):
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
    

def main():
    st.title("* Query-Doc-With-LightRAG *")
    
    indexs_list = ["contract_sample_1", "contract_sample_2", "multi_contracts"]  
    selected_index = st.sidebar.selectbox("Select-Index", indexs_list) 

    # Add a gap (line break)
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)
    selected_index_path = f"./{selected_index}"
    
    models = ["Azure-Openai-gpt-4o-mini", "meta-llama/Llama-3.2-1B"] 
    selected_model = st.sidebar.selectbox("Select a Model", models)
    st.sidebar.markdown("<br><br>", unsafe_allow_html=True)

    search_methods = ["naive","hybrid","mix"] 
    selected_search_method = st.sidebar.selectbox("Select a Search Method", search_methods)

    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
   
    user_input = st.chat_input("Enter your query")
    if user_input:
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role":"user", "content":user_input})

        with st.chat_message("assistant"):
            mass_holder = st.empty()
            if selected_model=="Azure-Openai-gpt-4o-mini":
                rag = lightrag_with_azureopenai_model(selected_index_path)
                lightrag_response = query_index(rag, user_input, selected_search_method)
            if selected_model=="meta-llama/Llama-3.2-1B":
                rag = lightrag_with_huggingface_model(selected_index_path)
                lightrag_response = query_index(rag, user_input, selected_search_method)

            mass_holder.markdown(lightrag_response)

        st.session_state.messages.append({"role":"assistant", "content":lightrag_response})
        

if __name__ == "__main__":
    main()


