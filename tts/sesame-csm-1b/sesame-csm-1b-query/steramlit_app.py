import streamlit as st
import requests
import json
import numpy as np
import base64
import soundfile as sf
import torchaudio
import torch
import os
from typing import List, Dict, Any
import tempfile
import time

st.set_page_config(page_title="Sesame TTS Interface", layout="wide")

def encode_audio(audio_tensor: torch.Tensor) -> str:
    audio_list = audio_tensor.tolist()
    audio_bytes = torch.tensor(audio_list).numpy().tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')

def prepare_context(speakers: List[int], transcripts: List[str], audio_files):
    context = []
    for transcript, speaker, audio_file in zip(transcripts, speakers, audio_files):
        # Create a temporary file to save the uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            tmp_file.write(audio_file.getvalue())
            tmp_path = tmp_file.name
        
        audio_tensor, sample_rate = torchaudio.load(tmp_path)
        audio_base64 = encode_audio(audio_tensor)
        
        # Clean up the temporary file
        os.unlink(tmp_path)

        context.append({"text": transcript, "speaker": speaker, "audio": audio_base64, "original_sample_rate": sample_rate})
    
    return context

def prepare_payload(text: str, speaker: int, context: List[Dict[str, Any]], max_audio_length_ms: int) -> Dict[str, Any]:
    return {
        "inputs": [
            {"name": "text", "datatype": "BYTES", "shape": [1], "data": [text]},
            {"name": "speaker", "datatype": "INT32", "shape": [1], "data": [speaker]},
            {"name": "context", "datatype": "BYTES", "shape": [len(context)], "data": [json.dumps(item) for item in context]},
            {"name": "max_audio_length_ms", "datatype": "INT32", "shape": [1], "data": [max_audio_length_ms]}
        ]
    }

def send_request(url: str, payload: Dict[str, Any]) -> requests.Response:
    return requests.post(url, headers={"Content-Type": "application/json"}, json=payload)

# App title and description
st.title("Sesame TTS Model Interface")
st.markdown("Upload audio samples, provide transcripts, and generate speech using the Sesame TTS model.")

# Triton server URL input
triton_url = st.text_input("Triton Server URL", value="<DEPLOYMENT_LINK>/v2/models/csm_1b/infer")

# Context inputs section
st.header("Context Inputs")

# Dynamic context inputs
num_contexts = st.number_input("Number of context examples", min_value=0, max_value=10, value=2)

speakers = []
transcripts = []
audio_files = []

for i in range(num_contexts):
    col1, col2, col3 = st.columns([1, 2, 2])
    
    with col1:
        speaker = st.number_input(f"Speaker ID #{i+1}", min_value=0, value=i % 2)
        speakers.append(speaker)
    
    with col2:
        transcript = st.text_area(f"Transcript #{i+1}", value=f"Example text for speaker {speaker}.", height=100)
        transcripts.append(transcript)
    
    with col3:
        audio_file = st.file_uploader(f"Audio file #{i+1}", type=["wav"])
        if audio_file:
            audio_files.append(audio_file)
            st.audio(audio_file)

# Generation parameters
st.header("Generation Parameters")

col1, col2 = st.columns(2)
with col1:
    gen_text = st.text_area("Text to generate", value="This is the text I want to convert to speech.", height=150)
with col2:
    gen_speaker = st.number_input("Speaker ID for generation", min_value=0, value=0)
    max_audio_length = st.number_input("Max audio length (ms)", min_value=1000, value=5000, step=1000)

# Generate button
generate_button = st.button("Generate Speech")

# Results section
st.header("Results")
result_placeholder = st.empty()
audio_placeholder = st.empty()

if generate_button:
    # Check if all audio files are uploaded
    if len(audio_files) != num_contexts:
        st.error(f"Please upload all {num_contexts} audio files.")
    else:
        with st.spinner("Preparing context and generating speech..."):
            # Prepare context
            context = prepare_context(speakers, transcripts, audio_files)
            
            # Prepare payload
            payload = prepare_payload(gen_text, gen_speaker, context, max_audio_length)
            
            # Send request
            try:
                start_time = time.time()
                response = send_request(triton_url, payload)
                inference_time = time.time() - start_time
                
                # Process response
                if response.status_code == 200:
                    result = response.json()
                    result_placeholder.success(f"Inference successful! (Time: {inference_time:.2f}s)")
                    
                    audio = np.array(result['outputs'][0]['data'], dtype=np.float32)
                    sample_rate = result['outputs'][1]['data'][0]
                    
                    # Create a temporary file for the audio
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                        sf.write(tmp_file.name, audio, sample_rate)
                        
                        # Display audio player
                        with open(tmp_file.name, "rb") as f:
                            audio_bytes = f.read()
                            audio_placeholder.audio(audio_bytes, format="audio/wav")
                        
                        # Offer download button
                        st.download_button(
                            label="Download generated audio",
                            data=audio_bytes,
                            file_name="generated_speech.wav",
                            mime="audio/wav"
                        )
                        
                        # Clean up
                        os.unlink(tmp_file.name)
                        
                    # Display audio details
                    st.text(f"Sample rate: {sample_rate} Hz")
                    st.text(f"Audio length: {len(audio)/sample_rate:.2f} seconds ({len(audio)} samples)")
                else:
                    result_placeholder.error(f"Request failed with status code {response.status_code}")
                    st.code(response.text)
            except Exception as e:
                result_placeholder.error(f"Error: {str(e)}")

# Add some helpful information at the bottom
st.markdown("---")
st.markdown("""
### Tips:
- Make sure the Triton Server URL is correct and accessible
- Upload WAV files for best compatibility
- Speaker IDs should match between context examples and generation if you want to maintain the same voice
""")
