# app.py
import streamlit as st
from src.infer import run_live

st.set_page_config(page_title="Sign → Text/Voice (Medical)", layout="centered")
st.title("(Medical)")

st.markdown("Perform one of the configured medical phrases. The app recognizes it, "
            "rewrites to clear clinical wording, and speaks it.")

threshold = st.slider("Confidence threshold", 0.30, 0.95, 0.60, 0.01)
say = st.checkbox("Speak output", value=True)

if "running" not in st.session_state:
    st.session_state.running = False

col1, col2 = st.columns(2)
start = col1.button("Start")
stop = col2.button("Stop")

video_box = st.empty()
status = st.empty()

from src.tts import speak

def frame_cb(img_rgb):
    # show the camera frame in the app
    video_box.image(img_rgb, channels="RGB", caption="Live camera")

def is_running():
    return st.session_state.running

if start:
    st.session_state.running = True
    status.info("Press Stop to end.")
    # stream predictions
    for pred, conf in run_live(threshold=threshold, frame_callback=frame_cb, running=is_running):
        status.success(f"Recognized: **{pred}** (p={conf:.2f})")
        if say:
            speak(pred)

if stop:
    st.session_state.running = False
    status.warning("Stopped.")