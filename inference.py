import torch
import sentencepiece as spm
from model import BigramLanguageModel  # import model.py for inference

def load_model(model_path='PoemGPT.pth'):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    
    checkpoint = torch.load(model_path, map_location=device)
    
    
    sp = spm.SentencePieceProcessor()
    sp.load("tokenizer_150.model")
    
    
    model = BigramLanguageModel(sp.get_piece_size()).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, sp, device

def generate_text(model, sp, prompt="", max_tokens=100, device='cpu'):
    
    if prompt:
        context = torch.tensor(sp.encode_as_ids(prompt), dtype=torch.long).unsqueeze(0).to(device)
    else:
        context = torch.zeros((1, 1), dtype=torch.long, device=device)
    
    # Generate text
    generated_ids = model.generate(context, max_new_tokens=max_tokens)[0].tolist()
    return sp.decode_ids(generated_ids)

if __name__ == "__main__":
    
    model, sp, device = load_model()
    
    while True:
        prompt = input("\nEnter a text ('q' for quit): ")
        if prompt.lower() == 'q':
            break
            
        generated_text = generate_text(model, sp, prompt, max_tokens=200, device=device)
        print("\Generated text:")
        print(generated_text)