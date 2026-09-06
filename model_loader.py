import json

import onnxruntime as ort
from huggingface_hub import hf_hub_download

MODEL_REPO = "chriamue/bird-species-classifier"


def load_bird_model():
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="model.onnx",
    )
    config_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="config.json",
    )

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    session = ort.InferenceSession(
        model_path,
        providers=["CPUExecutionProvider"],
    )

    return session, config["id2label"]
