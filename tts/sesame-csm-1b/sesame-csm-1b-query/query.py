import requests
import json
import numpy as np
import base64
import soundfile as sf
import torchaudio
import torch
import os
from typing import List, Dict, Any


def encode_audio(audio_tensor: torch.Tensor) -> str:
    audio_list = audio_tensor.tolist()
    audio_bytes = torch.tensor(audio_list).numpy().tobytes()
    return base64.b64encode(audio_bytes).decode('utf-8')


def prepare_context(speakers: List[int], transcripts: List[str], audio_paths: List[str]):
    context = []
    for transcript, speaker, audio_path in zip(transcripts, speakers, audio_paths):
        input_audio_path = os.path.expanduser(audio_path)
    
        audio_tensor, sample_rate = torchaudio.load(input_audio_path)
        audio_base64 = encode_audio(audio_tensor)

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

def main():
    # Configuration
    triton_url = "<DEPLOYMENT_LINK>/v2/models/csm_1b/infer"
    output_filename = "generated_speech.wav"

    speakers = [0, 1]
    transcripts = [
        "Hey how are you doing.",
        "All good, what about you?"
    ]
    audio_paths = [
        "~/test/utterance_0.wav",
        "~/test/utterance_1.wav"
    ]

    #Prepare context and payload
    context = prepare_context(speakers, transcripts, audio_paths)
    payload = prepare_payload("Me too, what have you been up to these days?", 0, context, 5000)

    # Send request
    response = send_request(triton_url, payload)

    # Process response
    if response.status_code == 200:
        result = response.json()
        print("Inference response received!")
        
        audio = np.array(result['outputs'][0]['data'], dtype=np.float32)
        sample_rate = result['outputs'][1]['data'][0]
        
        print(f"Sample rate: {sample_rate}")
        print(f"Audio shape: {len(audio)}")
        
        sf.write(output_filename, audio, sample_rate)
        print(f"Audio saved to {output_filename}")
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    main()
