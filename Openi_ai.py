import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

def build_llm():
    # Load environment variables from .env file
    load_dotenv()

    # Retrieve API key from environment
    openai_api_key = os.getenv("OPENAI_API_KEY")

    # Initialize the ChatOpenAI with the API key and model
    llm = ChatOpenAI(api_key=openai_api_key, model_name="gpt-4.1")

    # Rebuild the model to resolve Pydantic validation issues
    llm.model_rebuild()

    # Example usage: invoke the LLM with a prompt - optional for testing
    # response = llm.invoke("Explain how to use LangChain with OpenAI for LLM calls.")
    # print(response)

    return llm
