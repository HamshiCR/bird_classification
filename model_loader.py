from transformers import AutoImageProcessor, AutoModelForImageClassification

def load_bird_model():
    processor = AutoImageProcessor.from_pretrained("chriamue/bird-species-classifier")
    model = AutoModelForImageClassification.from_pretrained("chriamue/bird-species-classifier")
    return processor, model


