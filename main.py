# Trying whisper out 


# import torch, whisper
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model = whisper.load_model("medium.en").to(device)
# result = model.transcribe("./Test Audios/Recording (4).m4a")
# print(f' The text in video: \n {result["text"]}')


# # Sentiment Analysis
# from pprint import pprint

# from transformers import pipeline
# sentiment_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
# data = ["most likely"]
# res = sentiment_pipeline(data)
# pprint(res)




from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import torch


def predict_happiness(image_path, model, processor):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images = image, return_tensors = "pt")

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim = -1)
    top_prob, top_lbl = torch.topk(probs, 1)

    if top_lbl == 0:
        prediction = "Happy"
    else:
        prediction = "Sad"

    return prediction, top_prob.item()

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
model = ViTForImageClassification.from_pretrained("Ketanwip/happy_sad_model")
image_path = "./Recording/images.jpeg"


prediction, probability = predict_happiness(image_path, model, processor)


print(f"The face is predicted to be: {prediction} with a confidence of {probability:.2%}") 
