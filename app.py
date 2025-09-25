import os
import tempfile

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

### CHANGE 1: UNIFIED SECRETS MANAGEMENT ###
# This single block works for both local .streamlit/secrets.toml and cloud secrets
# You can now remove the .env file and the python-dotenv library.
try:
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    ELEVEN_API_KEY = st.secrets["ELEVEN_API_KEY"]
except KeyError:
    st.error("API keys not found. Please add them to your .streamlit/secrets.toml file or Streamlit Cloud secrets.")
    st.stop()

### CHANGE 2: EFFICIENT API CLIENTS (PERFORMANCE FIX) ###
# Cache the API clients to prevent re-initializing on every script run
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

### CHANGE 3: STATEFUL CHAT MODEL (CONVERSATION MEMORY) ###
# This ensures the model is created only once and remembers the conversation.
def initialize_chat_model():
    system_instruction = f"You are a helpful assistant. Respond ONLY in {st.session_state.language}."
    model = gen_ai_client.GenerativeModel(
        "gemini-1.5-flash",
        system_instruction=system_instruction
    )
    st.session_state.chat = model.start_chat()
    st.session_state.model_language = st.session_state.language

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "language" not in st.session_state:
    st.session_state.language = "Malayalam"
if "tts_enabled" not in st.session_state:
    st.session_state.tts_enabled = True

# Initialize or re-initialize the chat model if it doesn't exist or language has changed
if "chat" not in st.session_state or st.session_state.get("model_language") != st.session_state.language:
    initialize_chat_model()


# --- SIDEBAR ---
with st.sidebar:
    st.markdown("### AgriBuddy Settings")
    st.session_state.tts_enabled = st.toggle("Enable TTS", value=st.session_state.tts_enabled)
    st.selectbox("Language", ["Malayalam", "English", "Hindi", "Spanish"], key="language")
    if st.button("Clear Chat History"):
        st.session_state.history = []
        initialize_chat_model() # Re-create the chat object to clear its memory
        st.rerun()

# --- HELPERS ---
def transcribe_audio_file(audio_path):
    try:
        with open(audio_path, "rb") as audio_file:
            # The .convert() method doesn't take a model_id argument for STT
            transcript = eleven_client.speech_to_text.convert(file=audio_file)
        return transcript.text
    except Exception as e:
        st.error(f"Transcription error: {e}")
        return None

def tts_generate_bytes(text):
    # NOTE: You must have a Voice ID from your ElevenLabs Voice Lab
    VOICE_ID = "21m00Tcm4TlvDq8ikWAM" # Example Voice ID, replace with your own
    try:
        audio_bytes = eleven_client.generate(text=text, voice=VOICE_ID, model="eleven_multilingual_v2")
        return audio_bytes
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# --- UI ---
st.title("💬 AgriBuddy")

# Display chat history
chat_container = st.container()
for msg in st.session_state.history:
    with chat_container.chat_message(msg["role"]):
        st.write(msg["text"])

# Handle all inputs
user_prompt = st.chat_input("Type your message or use mic...")
mic_audio = mic_recorder(start_prompt="🎤", stop_prompt="⏹️", key="mic")

if mic_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_file:
        tmp_file.write(mic_audio["bytes"])
        user_prompt = transcribe_audio_file(tmp_file.name)
        os.remove(tmp_file.name)

if user_prompt:
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

    st.session_state.history.append({"role": "assistant", "text": bot_text})

    # Generate and play TTS if enabled
    if st.session_state.tts_enabled and bot_text:
        audio_bytes = tts_generate_bytes(bot_text)
        if audio_bytes:
            st.audio(audio_bytes, format="audio/mpeg")
