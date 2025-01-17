import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import AutoTokenizer
import sentencepiece as spm

#hyperparameters
batch_size = 64 #how many independent sequences will we process in parallel?
block_size = 256 #what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 3
dropout = 0.2
#------------------

torch.manual_seed(1337)

#read it in to inspect it
with open("ozdemir asaf last.txt", "r", encoding="utf-8") as f:
    text = f.read()

#Load SentencePiece tokenizer
sp = spm.SentencePieceProcessor()
sp.load('asaf_unigram_tokenizer.model')
vocab_size = sp.get_piece_size() 


def encode(text):
    return sp.encode_as_ids(text)

def decode(ids):
    return sp.decode_ids(ids)


data = torch.tensor(encode(text), dtype=torch.long)

#split up the dataset into train/val sets
n = int(0.9*len(data))
train_data = data[:n] # %90 will be train set
val_data = data[n:]

#data loading
def get_batch(split):
    #generate a small batch of data of inputs x and targets y
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    "one head of self attention"

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

        self.dropout = nn.Dropout(dropout)

    def forward(self ,x):
        B, T, C = x.shape
        k = self.key(x) #(B, T, C=head_size)
        q = self.query(x) #(B, T, C)
        #compute attention scores "affinities"
        wei = q @ k.transpose(-2, -1) * C**-0.5 #(B, T, C) @ (B, C, T) => (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) #(B, T, T)
        wei = F.softmax(wei, dim=-1) #(B, T, T)   (e ** -inf = 0)
        wei = self.dropout(wei)
        #perform the weighted aggregation of the values
        v = self.value(x)
        out = wei @ v #(B, T, T) @ (B, T, C) => (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    "multiple heads of self-attention in parallel"

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    "a simple linear layer followed by a non-linearity"


    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
        
    
    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    "Transformer block: communication followed by computation"

    def __init__(self, n_emb, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


#super simple bigram model
class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        # vocab_size artık tokenizer'dan geliyor
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) #final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        # idx and targets are both (B,T) tensor of integers
        tok_emb = self.token_embedding_table(idx) #(B, T, C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) #(T, C)
        x = tok_emb + pos_emb #(B, T, C)
        x = self.blocks(x) #(B, T, C)
        x = self.ln_f(x) #(B, T, C)
        logits = self.lm_head(x) #(B,T,vocab_size)


        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets) 
        
        return logits, loss


    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            #crop idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        
        return idx   
     
model = BigramLanguageModel()
m = model.to(device)

# print the number of parameters in the model
print(sum(p.numel() for p in m. parameters()) /1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):
    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print (f" step {iter}: train loss {losses ['train']:.4f}, val loss {losses ['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad (set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
context = torch.zeros((1, 1), dtype=torch. long, device=device)
print (decode(m.generate (context, max_new_tokens=500)[0].tolist()))

#Save the model

torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'tokenizer_path': "tokenizer_150.model",  
    'vocab_size': sp.get_piece_size()  
}, 'PoemGPT.pth')