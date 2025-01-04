from KeyVaultManager import KeyVaultManager

keyvault_manager = KeyVaultManager("kvsurewerx")

AZURE_OPENAI_API_KEY = keyvault_manager.get_secret("GPT-KEY")
AZURE_OPENAI_ENDPOINT = keyvault_manager.get_secret("GPT-BASE")
AZURE_OPENAI_DEPLOYMENT = 'gpt4ominiV1Pilot' #keyvault_manager.get_secret("GPT-MODEL-NAME") 
AZURE_EMBEDDING_DEPLOYMENT = keyvault_manager.get_secret("OPENAI-EMBEDDING-MODEL-NAME")
AZURE_OPENAI_API_VERSION = keyvault_manager.get_secret("GPT-VERSION")
