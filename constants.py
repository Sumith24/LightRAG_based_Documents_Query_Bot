from KeyVaultManager import KeyVaultManager

keyvault_manager = KeyVaultManager("kvsurewerx")

AZURE_OPENAI_API_KEY = keyvault_manager.get_secret("GPT-KEY")
AZURE_OPENAI_ENDPOINT = keyvault_manager.get_secret("GPT-BASE")
AZURE_OPENAI_DEPLOYMENT = 'LightRAG-gpt-4o-mini' #keyvault_manager.get_secret("GPT-MODEL-NAME") 
AZURE_EMBEDDING_DEPLOYMENT = "text-embedding-3-large" #keyvault_manager.get_secret("OPENAI-EMBEDDING-MODEL-NAME")
AZURE_OPENAI_API_VERSION = keyvault_manager.get_secret("GPT-VERSION")

HUGF_MODEL_NAME = "microsoft/Phi-3.5-mini-instruct", #"meta-llama/Llama-3.2-1B"
HUGF_EMBEDDING_MODEL = "thenlper/gte-small"
