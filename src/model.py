# src/model.py
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1,
                 max_seq_len: int = 256): # Added max_seq_len for PE
        super(Seq2SeqTransformer, self).__init__()
        
        self.emb_size = emb_size
        self.src_tok_emb = nn.Embedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = nn.Embedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout, max_len=max_seq_len)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=emb_size, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False) # batch_first=False default
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        
        decoder_layer = nn.TransformerDecoderLayer(d_model=emb_size, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout, batch_first=False)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.dropout_layer = nn.Dropout(dropout) # Added dropout

    def forward(self,
                src: torch.Tensor,
                tgt: torch.Tensor,
                src_mask: torch.Tensor | None = None, # Make explicit Optional
                tgt_mask: torch.Tensor | None = None,
                src_padding_mask: torch.Tensor | None = None,
                tgt_padding_mask: torch.Tensor | None = None,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        
        # PyTorch Transformer expects:
        # src: (S, N, E) where S is source sequence length, N is batch size, E is embedding dimension.
        # tgt: (T, N, E) where T is target sequence length.
        # src_mask: (S, S)
        # tgt_mask: (T, T)
        # src_padding_mask: (N, S)
        # tgt_padding_mask: (N, T)

        # Ensure inputs are sequence-first
        # If inputs are (N,S) or (N,T), permute them.
        # Assuming input tensors are already token IDs (integers)
        # Shape: src (seq_len_src, batch_size), tgt (seq_len_tgt, batch_size)

        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        
        # Apply dropout to embeddings
        src_emb = self.dropout_layer(src_emb)
        tgt_emb = self.dropout_layer(tgt_emb)

        memory = self.transformer_encoder(src_emb, src_mask, src_padding_mask)
        outs = self.transformer_decoder(tgt_emb, memory, tgt_mask, None, # memory_mask is None
                                        tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: torch.Tensor, src_mask: torch.Tensor | None = None, src_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        src_emb = self.positional_encoding(self.src_tok_emb(src) * math.sqrt(self.emb_size))
        src_emb = self.dropout_layer(src_emb) # Apply dropout
        return self.transformer_encoder(src_emb, src_mask, src_padding_mask)

    def decode(self, tgt: torch.Tensor, memory: torch.Tensor, tgt_mask: torch.Tensor | None = None, tgt_padding_mask: torch.Tensor|None=None) -> torch.Tensor:
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt) * math.sqrt(self.emb_size))
        tgt_emb = self.dropout_layer(tgt_emb) # Apply dropout
        return self.transformer_decoder(tgt_emb, memory, tgt_mask, memory_key_padding_mask=tgt_padding_mask) # memory_key_padding_mask for memory

def generate_square_subsequent_mask(sz: int, device: torch.device) -> torch.Tensor:
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

if __name__ == '__main__':
    # Example Usage
    device = torch.device("cpu")
    SRC_VOCAB_SIZE = 100
    TGT_VOCAB_SIZE = 100
    EMB_SIZE = 128 # Reduced for faster example
    NHEAD = 4      # Reduced
    FFN_HID_DIM = 128 # Reduced
    NUM_ENCODER_LAYERS = 2 # Reduced
    NUM_DECODER_LAYERS = 2 # Reduced
    MAX_SEQ_LEN = 64

    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE,
                                     NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, max_seq_len=MAX_SEQ_LEN).to(device)

    # Example input (batch_size=2, seq_len=10 for src, seq_len=12 for tgt)
    # Tensors should be LongTensor for token IDs
    # Shape: (seq_len, batch_size)
    src = torch.randint(0, SRC_VOCAB_SIZE, (10, 2)).long().to(device)
    tgt = torch.randint(0, TGT_VOCAB_SIZE, (12, 2)).long().to(device)

    # Create masks
    # src_mask and tgt_mask are for preventing attention to future tokens (tgt) or specific positions (src, rarely used)
    # padding_masks are for ignoring PAD tokens.
    
    # For decoder self-attention, prevent looking at future tokens
    tgt_seq_len = tgt.size(0)
    tgt_mask = generate_square_subsequent_mask(tgt_seq_len, device) # Shape (T,T)

    # Example padding masks (True where padded)
    # Shape (N, S) for src_padding_mask, (N, T) for tgt_padding_mask
    src_padding_mask = torch.tensor([[False]*8 + [True]*2, [False]*10], device=device).bool() # Batch 1 has 2 pads, Batch 2 has 0 pads
    tgt_padding_mask = torch.tensor([[False]*10 + [True]*2, [False]*12], device=device).bool()# Batch 1 has 2 pads, Batch 2 has 0 pads
    
    # Note: memory_key_padding_mask in forward should be same as src_padding_mask.
    logits = transformer(src, tgt, tgt_mask=tgt_mask, 
                         src_padding_mask=src_padding_mask, 
                         tgt_padding_mask=tgt_padding_mask,
                         memory_key_padding_mask=src_padding_mask) # Pass src_padding_mask as memory_key_padding_mask
    
    print("Output logits shape (seq_len_tgt, batch_size, tgt_vocab_size):", logits.shape)
    assert logits.shape == (tgt.size(0), tgt.size(1), TGT_VOCAB_SIZE)
    print("Transformer model seems to run.")