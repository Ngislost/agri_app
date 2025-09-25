import os
import tempfile
from dotenv import load_dotenv

import streamlit as st
import google.generativeai as gen_ai
from streamlit_mic_recorder import mic_recorder
from elevenlabs.client import ElevenLabs

# --- CONFIG ---
st.set_page_config(
    page_title="AgriBuddy",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

### FIX 1: ROBUST API KEY LOADING FOR LOCAL & CLOUD ###
# This logic works for both local development (using .env) and Streamlit Cloud (using Secrets)
try:
    # For Streamlit Cloud
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ELEVEN_API_KEY = st.secrets["ELEVEN_API_KEY"]
except KeyError:
    # For local development
    load_dotenv(".env")
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")

# Stop the app if keys are still not found
if not GOOGLE_API_KEY or not ELEVEN_API_KEY:
    st.error("API keys not found. Please add them to your Streamlit secrets or a local .env file.")
    st.stop()

### FIX 2: CACHE API CLIENTS FOR PERFORMANCE ###
@st.cache_resource
def get_google_client():
    gen_ai.configure(api_key=GOOGLE_API_KEY)
    return gen_ai

@st.cache_resource
def get_elevenlabs_client():
    return ElevenLabs(api_key=ELEVEN_API_KEY)

gen_ai_client = get_google_client()
eleven_client = get_elevenlabs_client()

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');
    body, .stApp { font-family: 'Inter', sans-serif !important; background-color: #FCFCEF !important; }
    h1, h2, h3, p, div { color: #2D2D2D !important; }
    [data-testid="stSidebar"] { background-color: #ECF3EC !important; border-right: 1px solid #D6E4D6; }
    [data-testid="stSidebar"] * { color: #2D2D2D !important; }
    .stChatMessage { border: 1px solid #E0E0E0; border-radius: 0.75rem !important; padding: 0.75rem; background-color: #f5f5f5 !important; margin-bottom: 0.5rem; color: #000 !important; }
    [data-testid="stChatInput"] textarea { background: #f5f5f5 !important; border: 1px solid #D6D6D6 !important; border-radius: 0.5rem !important; color: #000000 !important; }
    [data-testid="stChatInput"] textarea::placeholder { color: #555555 !important; }
    button, [data-testid="stSelectbox"] > div { border-radius: 0.5rem !important; border: 1px solid #D6D6D6 !important; }
</style>
""", unsafe_allow_html=True)


# --- SESSION STATE & MODEL INITIALIZATION ---
if "history" not in st.session_state:
    st.session_state.history = []
if "language" not in st.session_state:
    st.session_state.language = "Malayalam"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True

### FIX 3: EFFICIENT GEMINI MODEL HANDLING ###
# This ensures the model is created only once or when the language changes
def initialize_chat_model():
    system_instruction = f"You are a helpful assistant. Respond ONLY in {st.session_state.language}."
    model = gen_ai_client.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_instruction
    )
    st.session_state.chat = model.start_chat()
    # Track the language the current model was created for
    st.session_state.model_language = st.session_state.language

# Initialize the chat model if it doesn't exist or if the language has changed
if "chat" not in st.session_state or st.session_state.get("model_language") != st.session_state.language:
    initialize_chat_model()


# --- SIDEBAR ---
with st.sidebar:
    # Your sidebar code here... (no changes needed)
    st.markdown("## AgriBuddy Settings")
    st.session_state.tts_enabled = st.toggle("Enable TTS", value=st.session_state.tts_enabled)
    # When selectbox changes, the app reruns, and the model will be re-initialized above
    st.selectbox("Language", ["Malayalam", "English", "Hindi", "Spanish"], key="language")
    if st.button("Clear Chat History"):
        st.session_state.history = []
        # Also re-initialize the chat model to clear its internal history
        initialize_chat_model()
        st.rerun()

# --- HELPERS ---
### FIX 4: CORRECTED ELEVENLABS STT API CALL ###
def transcribe_audio_file(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(audio_bytes)
        audio_path = tmp_file.name

    try:
        with open(audio_path, "rb") as f:
            # The 'convert' method for STT does not take a 'model_id' argument
            transcript = eleven_client.speech_to_text.convert(file=f)
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None
    finally:
        os.remove(audio_path)

def tts_generate_bytes(text):
    try:
        # NOTE: You must have a Voice ID from your ElevenLabs Voice Lab
        VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Example Voice ID, replace with your own
        audio_bytes = eleven_client.generate(
            text=text,
            voice=VOICE_ID,
            model="eleven_multilingual_v2"
        )
        return audio_bytes
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# --- UI ---
st.title("💬 AgriBuddy")

# Display chat history
chat_container = st.container()
with chat_container:
    for msg in st.session_state.history:
        with st.chat_message(msg["role"]):
            st.write(msg["text"])

# Handle inputs
user_prompt = st.chat_input("Type your message or use mic...")
mic_audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key="mic")

if mic_audio:
    user_prompt = transcribe_audio_file(mic_audio["bytes"])

if user_prompt:
    # Add user message to history and display it
    st.session_state.history.append({"role": "user", "text": user_prompt})
    with chat_container.chat_message("user"):
        st.write(user_prompt)

    # Stream Gemini response
    with chat_container.chat_message("assistant"):
        placeholder = st.empty()
        bot_text = ""
        try:
            # Use the persistent chat session from st.session_state
            for chunk in st.session_state.chat.send_message(user_prompt, stream=True):
                if chunk.text:
                    bot_text += chunk.text
                    placeholder.markdown(bot_text + "▌")
            placeholder.markdown(bot_text)
        except Exception as e:
            bot_text = f"⚠️ Gemini error: {e}"
            placeholder.write(bot_text)

    # Add bot message to history
    st.session_state.history.append({"role": "assistant", "text": bot_text})

    # Generate and play TTS if enabled
    if st.session_state.tts_enabled and bot_text:
        audio_bytes = tts_generate_bytes(bot_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mpeg")

### FIX 5: REMOVED THE EXTRA '}' AT THE END OF THE FILE ###
