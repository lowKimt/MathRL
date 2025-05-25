# src/solver.py
import random
import torch
import torch.nn as nn
import torch.optim as optim
import sympy
from .data_models import MathProblem
from .tokenizer import Tokenizer # Use the new tokenizer
from .model import Seq2SeqTransformer, generate_square_subsequent_mask # Use the new model

class SolverAI:
    def __init__(self, device: torch.device, tokenizer: Tokenizer,
                 emb_size=128, nhead=4, num_encoder_layers=3, num_decoder_layers=3,
                 ffn_hid_dim=256, dropout=0.1, learning_rate=1e-4, max_seq_len=128):
        
        self.device = device
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        self.model = Seq2SeqTransformer(
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            emb_size=emb_size,
            nhead=nhead,
            src_vocab_size=self.tokenizer.vocab_size,
            tgt_vocab_size=self.tokenizer.vocab_size, # Assuming same vocab for src and tgt
            dim_feedforward=ffn_hid_dim,
            dropout=dropout,
            max_seq_len=self.max_seq_len
        ).to(self.device)

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate)
        # Ignore PAD token in loss calculation
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens["<PAD>"])
        
        self.train_history = {"loss": [], "accuracy": []} # Basic training log

    def _prepare_tensors(self, problem_text: str, solution_text: str | None = None):
        # Tokenize source (problem)
        src_tokens = self.tokenizer.tokenize(problem_text, add_sos=True, add_eos=True)
        if len(src_tokens) > self.max_seq_len:
            src_tokens = src_tokens[:self.max_seq_len-1] + [self.tokenizer.special_tokens["<EOS>"]] # Truncate
        
        src_tensor = torch.tensor(src_tokens, dtype=torch.long, device=self.device).unsqueeze(1) # (seq_len, batch_size=1)
        src_padding_mask = (src_tensor == self.tokenizer.special_tokens["<PAD>"]).transpose(0,1) # (batch_size=1, seq_len)


        if solution_text is not None:
            # For training: Tokenize target (solution)
            # Target input is <SOS> solution
            # Target output is solution <EOS>
            tgt_tokens_full = self.tokenizer.tokenize(solution_text, add_sos=True, add_eos=True)
            if len(tgt_tokens_full) > self.max_seq_len: # Truncate if too long
                 tgt_tokens_full = tgt_tokens_full[:self.max_seq_len-1] + [self.tokenizer.special_tokens["<EOS>"]]


            tgt_input_tokens = tgt_tokens_full[:-1] # All but <EOS>
            tgt_output_tokens = tgt_tokens_full[1:]  # All but <SOS>

            # Pad if necessary (though for batch_size=1 and careful generation, might not be initially)
            # This padding logic is simplified for batch_size=1. For batching, pad all to max_len_in_batch.
            
            # For single instance, we'll just use the length of the instance.
            # If we were batching, we'd pad all sequences in the batch to the same length.
            # Let's assume max_seq_len includes SOS/EOS.
            
            # Create tensors (seq_len, batch_size=1)
            tgt_input_tensor = torch.tensor(tgt_input_tokens, dtype=torch.long, device=self.device).unsqueeze(1)
            tgt_output_tensor = torch.tensor(tgt_output_tokens, dtype=torch.long, device=self.device).unsqueeze(1) # For loss calculation
            
            tgt_padding_mask = (tgt_input_tensor == self.tokenizer.special_tokens["<PAD>"]).transpose(0,1) # (batch_size=1, seq_len)
            
            return src_tensor, src_padding_mask, tgt_input_tensor, tgt_padding_mask, tgt_output_tensor
        
        return src_tensor, src_padding_mask, None, None, None


    def solve(self, problem: MathProblem) -> str:
        """Generates a solution using the Transformer model."""
        print(f"Solver (PyTorch): Attempting Problem ID {problem.id}: {problem.text}")
        self.model.eval() # Set model to evaluation mode

        src_tensor, src_padding_mask, _, _, _ = self._prepare_tensors(problem.text)
        
        # Encoder output (memory)
        memory = self.model.encode(src_tensor, src_padding_mask=src_padding_mask)
        
        # Decoder_input starts with <SOS> token
        # Shape: (current_seq_len, batch_size=1)
        ys = torch.ones(1, 1).fill_(self.tokenizer.special_tokens["<SOS>"]).type_as(src_tensor.data).to(self.device)
        
        generated_tokens = []

        for _ in range(self.max_seq_len -1): # Max output length
            tgt_mask = generate_square_subsequent_mask(ys.size(0), self.device) # (curr_seq_len, curr_seq_len)
            
            # No padding in `ys` during generation initially for batch_size=1
            # Tgt_padding_mask for decode would be all False if no padding in `ys`.
            # Or, if we fixed `ys` to always be max_len and padded, then it would be used.
            # For iterative generation for batch_size=1, it's simpler:
            out_decode = self.model.decode(ys, memory, tgt_mask=tgt_mask, tgt_padding_mask=None) # (curr_seq_len, 1, emb_size)
            
            # Get logits for the last token
            prob = self.model.generator(out_decode[-1, :, :]) # (1, vocab_size)
            
            _, next_word_idx = torch.max(prob, dim=1)
            next_word_idx_item = next_word_idx.item()
            
            if next_word_idx_item == self.tokenizer.special_tokens["<EOS>"]:
                break # End of sequence
            
            generated_tokens.append(next_word_idx_item)
            # Append predicted token to current target sequence to be fed in next step
            # ys shape (current_seq_len, 1)
            ys = torch.cat([ys, torch.ones(1, 1).type_as(src_tensor.data).fill_(next_word_idx_item).to(self.device)], dim=0)

        solution_str = self.tokenizer.detokenize(generated_tokens)
        print(f"Solver (PyTorch): Proposed solution for Problem ID {problem.id}: '{solution_str}'")
        return solution_str


    def update(self, problem: MathProblem, proposed_solution_str:str, evaluation_results: dict, solved_reward: float):
        """Trains the model on the given problem and its expected solution."""
        print(f"Solver (PyTorch): Received Solved Reward: {solved_reward:.2f} for Problem ID {problem.id}. Training...")
        self.model.train() # Set model to training mode

        # We train on the ground truth solution from problem.expected_solution_expr
        if problem.expected_solution_expr is None:
            print("Solver (PyTorch): No expected solution to train on. Skipping update.")
            return

        # Convert sympy expr to string for tokenizer
        expected_solution_text = sympy.printing.sstr(problem.expected_solution_expr)
        # If expected_solution_expr is a list of equations (e.g. from solve of quadratic)
        if isinstance(problem.expected_solution_expr, list):
             expected_solution_text = ", ".join([sympy.printing.sstr(eq) for eq in problem.expected_solution_expr])


        src_tensor, src_padding_mask, tgt_input_tensor, tgt_padding_mask, tgt_output_tensor = \
            self._prepare_tensors(problem.text, expected_solution_text)

        if tgt_input_tensor is None or tgt_output_tensor is None: # Should not happen if expected_solution_text is valid
            print("Solver (PyTorch): Failed to prepare target tensors for training. Skipping update.")
            return

        # Create masks
        # src_mask is typically not needed for standard encoder.
        # tgt_mask prevents attending to future tokens in the target sequence.
        tgt_seq_len = tgt_input_tensor.size(0)
        tgt_mask = generate_square_subsequent_mask(tgt_seq_len, self.device)

        self.optimizer.zero_grad()
        
        # Model output shape: (tgt_seq_len, batch_size, vocab_size)
        logits = self.model(src_tensor, tgt_input_tensor, 
                            tgt_mask=tgt_mask, 
                            src_padding_mask=src_padding_mask, 
                            tgt_padding_mask=tgt_padding_mask,
                            memory_key_padding_mask=src_padding_mask) # memory_key_padding_mask is src_padding_mask

        # Reshape for CrossEntropyLoss:
        # Logits: (N, C) where N is total number of tokens (across batch and seq_len), C is vocab_size
        # Target: (N)
        loss = self.criterion(logits.reshape(-1, logits.shape[-1]), tgt_output_tensor.reshape(-1))
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # Gradient clipping
        self.optimizer.step()

        self.train_history["loss"].append(loss.item())
        print(f"Solver (PyTorch): Trained on Problem ID {problem.id}. Loss: {loss.item():.4f}")

        # Placeholder for accuracy calculation (more involved, e.g. token-level accuracy)
        # For now, success is determined by the Evaluator based on the generated string.