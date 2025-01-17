import sentencepiece as spm

#Training tokenizer
spm.SentencePieceTrainer.train(
    input='ozdemir asaf last.txt',         
    model_prefix='tokenizer_150',  
    vocab_size=150,             
    model_type='unigram',        
    character_coverage=1.0,      
    bos_id=0, eos_id=1, unk_id=2, pad_id=3  
)
