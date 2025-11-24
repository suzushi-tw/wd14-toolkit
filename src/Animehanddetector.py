# from transformers import AutoModelForCausalLM, AutoTokenizer
# from PIL import Image
# from typing import Tuple, Dict, Any
# import torch

# class AnimeHandDetector:
#     def __init__(self):
#         # Enable CUDA optimizations
#         torch.backends.cuda.matmul.allow_tf32 = False
#         torch.backends.cudnn.allow_tf32 = False
#         torch.backends.cudnn.benchmark = True
        
#         # Load model with proper settings
#         self.model = AutoModelForCausalLM.from_pretrained(
#             "vikhyatk/moondream2",
#             revision="2025-01-09",
#             trust_remote_code=True,
#             torch_dtype=torch.float16,
#             device_map="cuda"
#         )
        
#         self.hand_prompt = """Pay close attention to the character's hands, and tell me if it's:
#         - an open palm with spread fingers
#         - a completely closed fist
#         - a semi-closed fist or hand holding something
#         Be specific in your description."""

#     @torch.inference_mode()
#     @torch.amp.autocast('cuda')
#     def detect_gesture(self, image: Image.Image, debug: bool = False) -> Tuple[Dict[str, Any], Dict[str, float], Image.Image]:
#         if image is None:
#             return {"error": "No image provided"}, {"confidence": 0.0}, None

#         response = self.model.query(image, self.hand_prompt)["answer"].lower()
        
#         detected_gesture = "no_hands_detected"
#         confidence = 0.0
        
#         # Enhanced gesture detection
#         if "open palm" in response or "open hand" in response or "spread fingers" in response:
#             detected_gesture = "open_palm"
#             confidence = 0.85
#         elif "completely closed" in response or "tight fist" in response:
#             detected_gesture = "fist"
#             confidence = 0.85
#         elif any(phrase in response for phrase in ["semi-closed", "holding", "partially closed", "loose fist"]):
#             detected_gesture = "semi_closed"
#             confidence = 0.85

#         return {
#             "num_hands_detected": 1 if detected_gesture != "no_hands_detected" else 0,
#             "gestures": [detected_gesture]
#         }, {"hand_1": confidence} if confidence > 0 else {"confidence": 0.0}, image