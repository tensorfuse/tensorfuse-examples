import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from diffusers import AutoPipelineForText2Image, FluxPipeline
from io import BytesIO

class TritonPythonModel:
    def initialize(self, args):
        """Load the Stable Diffusion model"""
        self.logger = pb_utils.Logger
        self.model_id = "black-forest-labs/FLUX.1-dev"

        try:
            # Load pipeline with fp16 optimization
            self.pipeline = FluxPipeline.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
            ).to("cuda")
            self.logger.log_info("Successfully loaded FLUX.1-dev model")

        except Exception as e:
            self.logger.log_error(f"Error initializing model: {str(e)}")
            raise

    def execute(self, requests):
        """Process requests and generate images"""
        responses = []

        for request in requests:
            try:
                # Get input prompt
                prompt = pb_utils.get_input_tensor_by_name(request, "PROMPT")
                prompt_str = prompt.as_numpy()[0].decode()

                
                # Generate image
                image = self.pipeline(
                     prompt=prompt_str,
                     num_inference_steps=25,
                     guidance_scale=7.5,
                     height=512,
                     width=512
                ).images[0]

                # Convert image to byte array
                img_byte_arr = BytesIO()
                image.save(img_byte_arr, format="PNG")
                img_np = np.frombuffer(img_byte_arr.getvalue(), dtype=np.uint8)

                # Create output tensor
                output_tensor = pb_utils.Tensor(
                    "GENERATED_IMAGE",
                    img_np
                )

                responses.append(pb_utils.InferenceResponse([output_tensor]))
                self.logger.log_info("Successfully generated image")

            except Exception as e:
                self.logger.log_error(f"Error processing request: {str(e)}")
                responses.append(pb_utils.InferenceResponse(error=str(e)))

        return responses

    def finalize(self):
        """Cleanup resources"""
        self.pipeline = None
        torch.cuda.empty_cache()