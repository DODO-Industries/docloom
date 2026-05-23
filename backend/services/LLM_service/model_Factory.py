import os
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from backend.config import envConfig
from backend.config.envConfig import setup_logger, log_service

logger = setup_logger("LLMModelFactory")

class ModelFactory:
    @staticmethod
    def get_model(provider=None, temperature=0.7):
        """
        Factory method to return a LangChain compatible LLM instance.
        Supported providers: GEMINI, LOCAL, GENAI
        """
        provider = provider or envConfig.DEFAULT_LLM_PROVIDER
        log_service(logger, f"Initializing LLM Provider: {provider} (temp={temperature})", "info")

        if provider == "GEMINI":
            if not envConfig.GEMINI_API_KEY:
                raise ValueError("GEMINI_API_KEY missing in .env.dev")
            return ChatGoogleGenerativeAI(
                model=envConfig.GEMINI_MODEL,
                google_api_key=envConfig.GEMINI_API_KEY,
                temperature=temperature
            )

        elif provider == "LOCAL":
            # LM Studio or local OpenAI-compatible server
            return ChatOpenAI(
                base_url=f"{envConfig.LOCAL_LM_HOST}/v1",
                api_key="not-needed", # Local usually doesn't require key
                model=envConfig.LOCAL_LM_MODEL_BASE,
                temperature=temperature
            )

        elif provider == "GENAI":
            # Custom GenAI Proxy (OpenAI Compatible)
            if not envConfig.GENAI_API_KEY:
                raise ValueError("GENAI_API_KEY missing in .env.dev")
            
            # Remove /v1 if it's already in the base_url or add if missing
            base_url = envConfig.GENAI_URL
            if not base_url.endswith("/v1") and not base_url.endswith("/v1/"):
                base_url = os.path.join(base_url, "v1") if not base_url.endswith("/") else base_url + "v1"

            return ChatOpenAI(
                base_url=base_url,
                api_key=envConfig.GENAI_API_KEY, # Keep for standard compat
                model=envConfig.GENAI_MODEL,
                temperature=temperature,
                default_headers={
                    "GENAI_KEY": envConfig.GENAI_API_KEY # Custom header for this proxy
                }
            )

        else:
            raise ValueError(f"Unsupported LLM Provider: {provider}")

# Example Usage:
# model = ModelFactory.get_model()
# response = model.invoke("Hello, who are you?")
