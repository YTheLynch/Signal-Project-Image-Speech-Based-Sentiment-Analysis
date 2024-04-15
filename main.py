import tkinter as tk
import torch, whisper
from pprint import pprint    
from transformers import pipeline
from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image



# Landing Interface
root = tk.Tk()
root.title("Sentiment Analysis")
root.geometry("400x300")
root.configure(bg="#ffdc26")

# Create a label
label = tk.Label(root, text="Which sentiment analysis to run?", font=("Arial", 18), bg="#ffdc26")
label.pack(pady=20)

# Define a global variable to store the selected sentiment type
sentiment_var = tk.StringVar(root)

def get_sentiment_type(selected_type):
    """
    This function is called when a button is pressed.
    It retrieves the selected sentiment analysis type from the button text.
    """
    global sentiment_var  # Access the global variable

    sentiment_var.set(selected_type)
    root.destroy()  # Close the window after selection


# Create a frame for horizontal button alignment
button_frame = tk.Frame(root, bg = "#ffdc26")
button_frame.pack(pady=50)

# Create buttons
button_speech = tk.Button(button_frame, text="Speech", command=lambda: get_sentiment_type("Speech"), width=10, height=2)
button_speech.pack(side=tk.LEFT)

button_image = tk.Button(button_frame, text="Image", command=lambda: get_sentiment_type("Image"), width=10, height=2)
button_image.pack(side=tk.LEFT, padx=20)  # Add padding between buttons

button_both = tk.Button(button_frame, text="Both", command=lambda: get_sentiment_type("Both"), width=10, height=2)
button_both.pack(side=tk.LEFT)

# Run the main loop
root.mainloop()

# After the window closes, the selected sentiment type is stored in sentiment_var
selected_sentiment = sentiment_var.get()


def getAudio():
    with open("./utility/captureaudio.py") as file:
        exec(file.read())
    
def getImage():
    with open("./utility/captureimage.py") as file:
        exec(file.read())


def analyseAudio():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = whisper.load_model("medium.en").to(device)
    result = model.transcribe("./Captures/recording0.wav")
    print(f' The text in audio: \n {result["text"]}')
    
    sentiment_pipeline = pipeline("text-classification", model='bhadresh-savani/distilbert-base-uncased-emotion', top_k = None)
    data = result["text"]
    res = sentiment_pipeline(data)
    pprint(res)


def analyseImage():
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



if (selected_sentiment == "Audio"):
    getAudio()
    analyseAudio()
elif (selected_sentiment == "Image"):
    pass
else:
    pass

