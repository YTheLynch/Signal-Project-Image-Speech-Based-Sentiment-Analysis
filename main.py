# Trying whisper out 


import whisper
model = whisper.load_model("base")
result = model.transcribe("./Recording (4).m4a", fp16 = False)
print(f' The text in video: \n {result["text"]}')

