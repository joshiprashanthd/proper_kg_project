from openai import OpenAI
from transformers import AutoTokenizer, AutoModelForCausalLM
from groq import Groq
from dotenv import load_dotenv
import os
import requests
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_result
)

load_dotenv()

class BaseModel:
    def __init__(self, model_name, device):
        self.device = device
        self.model_name = model_name

    def generate_text(self, prompt: str, max_tokens=512, temperature=0.7):
        raise NotImplementedError
    
class OpenAIModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(8)) 
    def generate_text(self, prompt, max_tokens=512, temperature=0.7, structured_format=None):
        request_params = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "seed": 42,
            "max_completion_tokens": max_tokens,
        }
        
        if structured_format:
            response = self.client.beta.chat.completions.parse(
                **request_params,
                response_format=structured_format
            )
            return response.choices[0].message.parsed
        
        response = self.client.chat.completions.create(**request_params)
        return response.choices[0].message.content
    

class LlamaModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.model.generation_config.pad_token_id = self.tokenizer.eos_token_id
    
    def generate_text(self, prompt, max_tokens=512, temperature=0.7, max_length=1024):
        inputs = self.tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=max_tokens)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        outputs = self.model.generate(inputs['input_ids'], attention_mask=inputs['attention_mask'], max_length=max_length)
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
class GroqModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    def generate_text(self, prompt, max_tokens=512, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content

class HFApiModel(BaseModel):
    def __init__(self, model_name, device):
        super().__init__(model_name, device)
        self.api_url = f"https://api-inference.huggingface.co/models/{self.model_name}"
        self.api_key = os.getenv("HF_API_KEY")
    
    def _try(self, api_key, prompt, max_tokens, temperature):
        headers = {"Authorization": f"Bearer {api_key}"}
        response = requests.post(
            self.api_url,
            headers=headers,
            json={
                "inputs": prompt,
                "options": {"use_cache": True},
                "parameters": {"max_new_tokens": max_tokens, "temperature": temperature, "return_full_text": False}
            }
        )

        if response.status_code == 200:
            response_data = response.json()
            if isinstance(response_data, list) and 'generated_text' in response_data[0]:
                return response_data[0]['generated_text'].strip()
        return None
    
    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(8), retry=retry_if_result(lambda x: x is None)) 
    def generate_text(self, prompt, max_tokens=512, temperature=0.7):
        return self._try(self.api_key, prompt, max_tokens, temperature)
