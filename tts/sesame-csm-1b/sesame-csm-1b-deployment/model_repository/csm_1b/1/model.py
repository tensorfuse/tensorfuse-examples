import json
import os
import triton_python_backend_utils as pb_utils
import torch
import numpy as np
import sys
import base64
import torchaudio

class TritonPythonModel:
    def initialize(self, args):
        self.logger = pb_utils.Logger

        self.model_dir = args['model_repository']
        self.model_id = "sesame/csm-1b"

        
        # Import after installation
        from generator import load_csm_1b
        from huggingface_hub._login import _login
        _login(token=os.environ.get("HUGGING_FACE_HUB_TOKEN"), add_to_git_credential=True)

        
        # Load the model
        try:
            self.generator = load_csm_1b(device="cuda" if torch.cuda.is_available() else "cpu")
            self.logger.log_info("Successfully loaded Sesame 1B")
        except Exception as e:
            self.logger.log_error(f"Error initializing model: {str(e)}")
            raise        
        print("CSM-1B model loaded successfully", file=sys.stderr)


    def execute(self, requests):
        responses = []

        from generator import Segment
        
        for request in requests:
            # Extract inputs
            text_tensor = pb_utils.get_input_tensor_by_name(request, "text")
            voice_tensor = pb_utils.get_input_tensor_by_name(request, "speaker")
            context_tensor = pb_utils.get_input_tensor_by_name(request, "context")
            max_size_tensor = pb_utils.get_input_tensor_by_name(request, "max_audio_length_ms")

            text_data = text_tensor.as_numpy()      # shape: (1,)
            voice_data = voice_tensor.as_numpy()    # shape: (1,)
            context_data = context_tensor.as_numpy() # shape: (N,)
            max_size_data = max_size_tensor.as_numpy() # shape: (1,)

            text_value = text_data[0]
            if isinstance(text_value, bytes):
                text_value = text_value.decode("utf-8")

            voice_value = int(voice_data[0])

            context_list = []
            for c in context_data:
                context_item = json.loads(c.decode("utf-8") if isinstance(c, bytes) else str(c))
                audio_tensor = context_item['audio']
                sample_rate = context_item['original_sample_rate']
                
                
                audio_bytes = base64.b64decode(audio_tensor)
                audio_array = np.frombuffer(audio_bytes, dtype=np.float32)
                audio = torch.from_numpy(audio_array)
                audio = torchaudio.functional.resample(audio.squeeze(0), orig_freq=sample_rate, new_freq=self.generator.sample_rate)
                
                segment = Segment(
                    text=context_item['text'],
                    speaker=context_item['speaker'],
                    audio=audio
                )
                context_list.append(segment)

            max_size_value = int(max_size_data[0])

            
            # Generate audio
            audio = self.generator.generate(
                text=text_value,
                speaker=voice_value,
                context=context_list,
                max_audio_length_ms=max_size_value
            )
            
            # Create output tensors
            audio_tensor = pb_utils.Tensor("audio", audio.cpu().numpy())
            sample_rate_tensor = pb_utils.Tensor("sample_rate", 
                                            np.array([self.generator.sample_rate], dtype=np.int32))
            
            # Create and append response
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[audio_tensor, sample_rate_tensor]
            )
            responses.append(inference_response)
        
        return responses

    def finalize(self):
        print("Unloading CSM-1B model", file=sys.stderr)   
        self.generator = None
        torch.cuda.empty_cache() 
