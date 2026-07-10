from langchain_ollama import OllamaEmbeddings


def get_embedding_function():
    # Use Ollama embeddings by default. This avoids AWS Bedrock credentials.
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    # If you want to use AWS Bedrock instead, install langchain-aws and uncomment below:
    # from langchain_aws import BedrockEmbeddings
    # embeddings = BedrockEmbeddings(
    #     credentials_profile_name="default", region_name="us-east-1"
    # )
    return embeddings
