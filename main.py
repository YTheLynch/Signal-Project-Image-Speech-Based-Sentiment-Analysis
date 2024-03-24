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


