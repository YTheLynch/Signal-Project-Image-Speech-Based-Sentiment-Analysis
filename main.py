import tkinter as tk
import torch, whisper
from pprint import pprint    
from transformers import pipeline
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
from utility import captureaudio, captureimage


root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x300")
root.configure(bg="#ffdc26")

label = tk.Label(root, text="Which sentiment analysis to run?", font=("Arial", 18), bg="#ffdc26")
label.pack(pady=20)

sentiment_var = tk.StringVar(root)

def get_sentiment_type(selected_type):
    global sentiment_var  

    sentiment_var.set(selected_type)
    root.destroy()  


button_frame = tk.Frame(root, bg = "#ffdc26")
button_frame.pack(pady=50)

button_speech = tk.Button(button_frame, text="Speech", command=lambda: get_sentiment_type("Speech"), width=10, height=2)
button_speech.pack(side=tk.LEFT)

button_image = tk.Button(button_frame, text="Image", command=lambda: get_sentiment_type("Image"), width=10, height=2)
button_image.pack(side=tk.LEFT, padx=20) 

button_both = tk.Button(button_frame, text="Both", command=lambda: get_sentiment_type("Both"), width=10, height=2)
button_both.pack(side=tk.LEFT)


root.mainloop()


selected_sentiment = sentiment_var.get()

resultAudio = [0, 0]
resultImage = [0, 0]

def analyseAudio():
    captureaudio.main()
    print("Analysing")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model("medium.en").to(device)
    result = model.transcribe("./Captures/recording0.wav", fp16 = False)
    print(f' The text in audio: \n {result["text"]}')
    
    sentiment_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
    data = result["text"]
    res = sentiment_pipeline(data)
    # pprint(res)  
    for i in res[0]:
        if (i['label'] in ['anger', 'fear', 'sadness']):
            resultAudio[0] += i['score']
        elif (i['label'] in ['joy', 'love', 'surprise']):
            resultAudio[1] += i['score']

    if (resultAudio[0] > resultAudio[1]):
        print(f"Speech Based Happiness Rating: Sad, Probability: {resultAudio[0]}%")
    else:
        print(f"Speech Based Happiness Rating: Happy, Probability: {resultAudio[1]}%")

def analyseImage():
    counter = captureimage.main()
    print("Analysing")
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
    for i in range(counter):
        image_path = f"./Captures/captured_image{i}.jpg"
        prediction, probability = predict_happiness(image_path, model, processor)

        if (prediction == "Sad"):
            resultImage[0] += probability
        else:
            resultImage[1] += probability


    if (resultImage[0] > resultImage[1]):
        print(f"Image Based Happiness Rating: Sad, Probability = {float(resultImage[0])/counter:.2f}")
    else:
        print(f"Image Based Happiness Rating: Happy, Probability = {float(resultImage[1])/counter:.2f}")



if (selected_sentiment == "Speech"):
    analyseAudio()
elif (selected_sentiment == "Image"):
    analyseImage()
elif (selected_sentiment == "Both"):
    analyseAudio()
    analyseImage()