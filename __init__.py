import os
import sys
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np

from src.pipeline import process_image


class PhotoVerification:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",)
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "verify"
    CATEGORY = "Fotobudka"

    def verify(self, image):
        # Convert tensor to PIL Image and save to temp file
        pil_image = Image.fromarray((image[0].cpu().numpy() * 255).astype(np.uint8))

        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
            pil_image.save(tmp_path)

        try:
            result = process_image(Path(tmp_path))
        finally:
            os.remove(tmp_path)

        if result.get("status") == "ok":
            return (image,)
        else:
            raise Exception(f"Photo verification failed: {result.get('message', 'Unknown error')}")

NODE_CLASS_MAPPINGS = {
    "PhotoVerification": PhotoVerification
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "PhotoVerification": "Photo Verification"
}
