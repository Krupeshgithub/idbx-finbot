import os
from transformers import pipeline

def download_model():
    print("Pre-downloading ProsusAI/finbert model into Docker image...")
    # This will download the model files and save them in the default HF_HOME cache dir
    # inside the docker container (~/.cache/huggingface/hub)
    pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert", 
        tokenizer="ProsusAI/finbert"
    )
    print("Model downloaded and cached successfully.")

if __name__ == "__main__":
    download_model()
