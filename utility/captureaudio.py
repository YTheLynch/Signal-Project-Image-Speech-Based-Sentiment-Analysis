import tkinter as tk
from PIL import Image, ImageTk 
import sounddevice as sd
from scipy.io.wavfile import write
import asyncio

async def record_audio_main(duration):
    freq = 44100
    recording = sd.rec(int(duration * freq), samplerate=freq, channels=2)
    print("Recording...")
    sd.wait()
    write("./Recording/recording0.wav", freq, recording)
    print(f"Recorded for {duration} seconds...")
    

async def record_audio(duration):
    asyncio.create_task(record_audio_main(duration))   


def create_menu(button):
    menu = tk.Menu(button, tearoff=0)
    menu.add_radiobutton(label="5 Seconds", command=lambda: asyncio.run(record_audio(5)))
    menu.add_radiobutton(label="10 Seconds", command=lambda: asyncio.run(record_audio(10)))
    menu.add_radiobutton(label="20 Seconds", command=lambda: asyncio.run(record_audio(20)))
    menu.add_radiobutton(label="30 Seconds", command=lambda: asyncio.run(record_audio(30)))
    menu.add_radiobutton(label="1 Minute", command=lambda: asyncio.run(record_audio(60)))
    menu.add_radiobutton(label="2 Minutes", command=lambda: asyncio.run(record_audio(120)))
    menu.add_radiobutton(label="5 Minutes", command=lambda: asyncio.run(record_audio(300)))
    menu.add_radiobutton(label="10 Minutes", command=lambda: asyncio.run(record_audio(600)))
    return menu

# Create main window
root = tk.Tk()
root.title("Record Audio")
root.geometry("400x300")
root.configure(bg="#ffdc26")

# Load recording icon image (replace with your image path)
image_path = "./icons/record-icon.png"
icon = ImageTk.PhotoImage(Image.open(image_path).resize((32, 32)))

# Create label and button
label = tk.Label(root, text="Record Audio", font=("Arial", 16), bg="#ffdc26")
recording_label = tk.Label(root, text="Start recording", bg="#ffdc26")  # Label for recording status

button = tk.Button(root, image=icon, command=lambda: create_menu(button).tk_popup())

def get_button_position(event):
    x = button.winfo_rootx()
    y = button.winfo_rooty()
    create_menu(button).tk_popup(x, y)

# Bind the event to get button position on click
button.bind("<Button-1>", get_button_position)

# Layout elements
label.pack(pady=20)
recording_label.pack()
button.pack()

# Run the main loop
root.mainloop()
