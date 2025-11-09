from openai import AzureOpenAI
import os
from dotenv import load_dotenv

load_dotenv()



model_name = "gpt-4o"
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4o")
api_version = "2024-12-01-preview"

try:
    client = AzureOpenAI(
        api_version=api_version,
        azure_endpoint=endpoint,
        api_key=subscription_key,
    )
except Exception as e:
    print(f"Failed to initialize AzureOpenAI client: {e}")
    client = None



