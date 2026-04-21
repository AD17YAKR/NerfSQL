import os
import yaml
from langchain_groq import ChatGroq

def get_llm() -> ChatGroq:
    with open("configs/model.yaml") as f:
        cfg = yaml.safe_load(f)
    return ChatGroq(
        model=cfg["model"],
        temperature=cfg["temperature"],
        max_tokens=cfg["max_tokens"],
        api_key=os.environ["GROQ_API_KEY"],
    )
