# src/tokenizer.py
import sympy

class Tokenizer:
    def __init__(self):
        self.special_tokens = {
            "<PAD>": 0,  # Padding
            "<SOS>": 1,  # Start of Sequence
            "<EOS>": 2,  # End of Sequence
            "<UNK>": 3   # Unknown character
        }
        
        # Extended vocabulary for math expressions
        self.allowed_chars = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ+-*/=()., ^: "
        
        self.char_to_idx = {**self.special_tokens}
        self.idx_to_char = {idx: char for char, idx in self.special_tokens.items()}
        
        for char in self.allowed_chars:
            if char not in self.char_to_idx: # Avoid overwriting special tokens if they overlap
                idx = len(self.char_to_idx)
                self.char_to_idx[char] = idx
                self.idx_to_char[idx] = char
        
        self.vocab_size = len(self.char_to_idx)

    def tokenize(self, text: str, add_sos=True, add_eos=True) -> list[int]:
        if isinstance(text, (sympy.Expr, sympy.Eq)): # Convert sympy expressions to string
            text = sympy.printing.sstr(text) # Use sstr for a more canonical representation

        tokens = []
        if add_sos:
            tokens.append(self.special_tokens["<SOS>"])
        
        for char in text:
            tokens.append(self.char_to_idx.get(char, self.special_tokens["<UNK>"]))
            
        if add_eos:
            tokens.append(self.special_tokens["<EOS>"])
        return tokens

    def detokenize(self, token_ids: list[int], strip_special=True) -> str:
        chars = []
        for token_id in token_ids:
            if strip_special and token_id in self.special_tokens.values():
                if token_id == self.special_tokens["<EOS>"]: # Stop at EOS if stripping
                    break 
                if token_id == self.special_tokens["<PAD>"] or \
                   token_id == self.special_tokens["<SOS>"]:
                    continue
            chars.append(self.idx_to_char.get(token_id, "?")) # Use ? for out-of-vocab during detokenization
        return "".join(chars)

if __name__ == '__main__':
    tokenizer = Tokenizer()
    print(f"Vocabulary Size: {tokenizer.vocab_size}")
    print(f"Char to Idx: {tokenizer.char_to_idx}")
    
    example_problem = "2*x + 3 = 7"
    tokenized = tokenizer.tokenize(example_problem)
    print(f"Tokenized '{example_problem}': {tokenized}")
    detokenized = tokenizer.detokenize(tokenized)
    print(f"Detokenized: '{detokenized}'")

    example_solution_expr = sympy.Eq(sympy.symbols('x'), 2)
    tokenized_sol = tokenizer.tokenize(example_solution_expr)
    print(f"Tokenized Sympy '{example_solution_expr}': {tokenized_sol}")
    detokenized_sol = tokenizer.detokenize(tokenized_sol)
    print(f"Detokenized Sympy: '{detokenized_sol}'")