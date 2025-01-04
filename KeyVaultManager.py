from azure.identity import DefaultAzureCredential 
from azure.keyvault.secrets import SecretClient
import logging
 
class KeyVaultManager:
    def __init__(self, keyvault_name):
        self.keyvault_name = keyvault_name
        self.credential = DefaultAzureCredential()
        #Vault URI
        self.secret_client = SecretClient(
            vault_url=f"https://{keyvault_name}.vault.azure.net", credential=self.credential)

    def get_secret(self, secretkey):
        try:
            secret = self.secret_client.get_secret(secretkey).value
            return secret
        except Exception as e:
            logging.exception(e)
            return None