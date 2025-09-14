# src/tts.py
def speak(text: str):
    try:
        import pyttsx3
        e = pyttsx3.init()
        e.say(text); e.runAndWait()
    except Exception:
        print("[TTS]", text)