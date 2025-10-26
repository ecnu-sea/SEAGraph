import torch
import torch.nn.functional as F
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from vllm import LLM
from vllm.sampling_params import SamplingParams

class SentenceEncoder:
    def __init__(self, device):
        self.dim = 768
        self.model = SentenceTransformer('all-MiniLM-L6-v2', cache_folder='.cache/huggingface/hub')
        self.model.to(device)

    def encode(self, input_texts: list[str]) -> Tensor:
        all_embeddings = self.model.encode(input_texts, convert_to_tensor=True)
        all_embeddings = F.normalize(all_embeddings, dim=1)
        return all_embeddings.cpu()
    

class GenerativeModel:
    def __init__(self, device):
        self.device = device
        model_name = "./Ministral-8B-Instruct-2410"
        self.sampling_params = SamplingParams(max_tokens=8192)
        self.model = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    def encode(self, messages):
        with torch.no_grad():
            outputs = self.model.chat(messages, sampling_params=self.sampling_params)
            torch.cuda.empty_cache()
        return outputs[0].outputs[0].text.strip()
