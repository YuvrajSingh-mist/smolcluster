import requests
from typing import Dict
import threading

lock = threading.Lock()
rollouts = {}

with open("configs/inference/model_config_inference.yaml") as f:
    model_config = yaml.safe_load(f)

def query(url: str, payload: Dict, prompt: str, decoding_strategy: str, max_tokens: int) -> str:
    
   
    response = requests.post(url, json=payload)
    return response.json()

def handle_worker(url: str, payload: Dict, prompt: str, worker_rank: int, decoding_strategy: str, max_tokens: int, rollouts: Dict[int, str], lock: threading.Lock) -> str:
    
    response = query(url, payload, prompt, decoding_strategy, max_tokens)
    
    with lock:
        rollouts[worker_rank] = response["generated_text"]
  
  
def generate_rollouts(prompt: str, decoding_strategy: str, max_tokens: int, rollouts: Dict[int, str], lock: threading.Lock) -> list[str]:
    threads = []  # Store thread references
    
    for worker_rank in range(3):
        url = "http://localhost:8080/query" 
        
        payload = {
            "text": prompt,
            "worker_rank": worker_rank,
            "max_tokens": max_tokens,
            "decoding_strategy": decoding_strategy,
        }
         
        thread = threading.Thread(
            target=handle_worker, 
            args=(url, payload, prompt, worker_rank, decoding_strategy, max_tokens, rollouts, lock)
        )  
        thread.start()
        threads.append(thread)  # Keep reference
    
    # Wait for all threads to complete before returning
    for thread in threads:
        thread.join()
        
def main():

    
    prompt = "What is the capital of France?"
    decoding_strategy = "top_p"
    max_tokens = 256
    
    main_thread = threading.Thread(target=generate_rollouts, args=(prompt, decoding_strategy, max_tokens, rollouts, lock))
    main_thread.start()
    main_thread.join()
    
    
    
if __name__ == "__main__":
    main()