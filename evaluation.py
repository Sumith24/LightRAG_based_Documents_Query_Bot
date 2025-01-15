import os
import pandas as pd
from lightrag import QueryParam
from sentence_transformers import SentenceTransformer
from lightrag_approaches import lightrag_with_azureopenai_model
from constants import HUGF_EMBEDDING_MODEL

device = 'cpu'
INDEXS = ['contract_sample_1', 'contract_sample_2']
# files_list = ['contract_sample_1_results', 'contract_sample_2_results', 'multi_contracts_before_eval_results']
emb_model = SentenceTransformer(HUGF_EMBEDDING_MODEL, device=device)

def get_similarity_score(text_a, text_b):
    groundtruth_embeddings = emb_model.encode(text_a).tolist()
    lightrag_resp_embeddings = emb_model.encode(text_b).tolist()
    similarity_score = emb_model.similarity(groundtruth_embeddings, lightrag_resp_embeddings)[0][0]
    return similarity_score


def get_response_with_naive(query, light_rag):
    return light_rag.query(query, param=QueryParam(mode="naive"))

# def get_response_with_local(query, light_rag):
#     return light_rag.query(query, param=QueryParam(mode="local"))

# def get_response_with_global(query, light_rag):
#     return light_rag.query(query, param=QueryParam(mode="global"))

# def get_response_with_hybrid(query, light_rag):
#     return light_rag.query(query, param=QueryParam(mode="hybrid"))

# def get_response_with_mix(query, light_rag):
#     return light_rag.query(query, param=QueryParam(mode="mix"))


def main():
    df_test = pd.read_csv("data/Contract_TestQuestions_2.csv", encoding="cp1252")
    # import pdb;pdb.set_trace()
    for index_storage in INDEXS:
        rag = lightrag_with_azureopenai_model(index_storage)
        print(f'\nProcessing {index_storage}:-----------------------------')
        for ind, row in df_test.iterrows():
            if row['Contract'].split('.')[0].split('-')[-1] in index_storage.split('_'):
                lightrag_response = get_response_with_naive(row['Question'], rag)
                df_test.at[ind, 'LightRAG_naive_response'] = lightrag_response
                similarity_score = get_similarity_score(row['Answer'], lightrag_response)
                df_test.at[ind, 'lightRag_similarity_scores'] = similarity_score
                if similarity_score>=90:
                    df_test.at[ind, 'LightRAG_evaluation_result'] = "Yes"
                else:
                    df_test.at[ind, 'LightRAG_evaluation_result'] = "No"

        df_test.to_csv(f'evaluations/{index_storage}_first_backup.csv', index=False)

        # df_test['LightRAG_naive_response'] = df_test['question'].apply(lambda question: get_response_with_naive(question, rag))
        # df_test['similarity_scores'] = df_test.apply(lambda row: get_similarity_score(row), axis=1)
        # df_test['LightRAG_evaluation_result'] = df_test['similarity_scores'].apply(lambda x: "Yes" if x>=90 else "No")

    df_test.to_csv(f'evaluations/LightRAG_evaluations.csv', index=False)


if __name__=="__main__":
    main()