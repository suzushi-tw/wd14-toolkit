import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration
import os

class JoyCaptioner:
    def __init__(self, model_id="fancyfeast/llama-joycaption-alpha-two-hf-llava"):
        self.model_id = model_id
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = None
        self.processor = None

    def load_model(self):
        if self.model is not None:
            return

        print(f"Loading JoyCaption model: {self.model_id}")
        self.processor = AutoProcessor.from_pretrained(self.model_id)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device
        )
        self.model.eval()
        print("JoyCaption model loaded.")

    def unload_model(self):
        if self.model is not None:
            del self.model
            del self.processor
            torch.cuda.empty_cache()
            self.model = None
            self.processor = None
            print("JoyCaption model unloaded.")

    def predict(self, image, tags=None):
        if self.model is None:
            self.load_model()

        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        prompt_text = "Write a long descriptive caption for this image in a formal tone."
        if tags:
            prompt_text += f" Include these elements in the description: {tags}"

        convo = [
            {
                "role": "system",
                "content": "You are a helpful image captioner.",
            },
            {
                "role": "user",
                "content": prompt_text,
            },
        ]

        convo_string = self.processor.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)

        inputs = self.processor(text=[convo_string], images=[image], return_tensors="pt").to(self.device)
        inputs['pixel_values'] = inputs['pixel_values'].to(torch.bfloat16)

        with torch.no_grad():
            generate_ids = self.model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=True,
                use_cache=True,
                temperature=0.6,
                top_p=0.9,
            )[0]

        generated_ids_trimmed = generate_ids[inputs['input_ids'].shape[1]:]
        caption = self.processor.tokenizer.decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        
        return caption.strip()
