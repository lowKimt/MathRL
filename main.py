# main.py
import sympy

from src.challenger import ChallengerAI
from src.solver import SolverAI # This will be the PyTorch SolverAI
from src.evaluator import Evaluator
from src.training_orchestrator import TrainingOrchestrator
from src.utils import get_device # Import device utility
from src.tokenizer import Tokenizer # Import tokenizer

# --- Configuration for PyTorch SolverAI ---
# Model Hyperparameters (adjust as needed)
EMB_SIZE = 128          # Embedding dimension
NHEAD = 4               # Number of attention heads
FFN_HID_DIM = 256     # Feedforward network hidden dimension
NUM_ENCODER_LAYERS = 3  # Number of encoder layers
NUM_DECODER_LAYERS = 3  # Number of decoder layers
DROPOUT = 0.1           # Dropout rate
MAX_SEQ_LEN = 128       # Max sequence length for problems and solutions

# Training Hyperparameters
LEARNING_RATE = 1e-4    # Learning rate for the optimizer
NUM_ITERATIONS = 100    # Number of training iterations (problems generated and attempted)

if __name__ == "__main__":
    # 1. Get Device
    device = get_device()

    # 2. Initialize Tokenizer
    tokenizer = Tokenizer()
    print(f"Initialized Tokenizer with vocab size: {tokenizer.vocab_size}")

    # 3. Initialize Components
    challenger_ai = ChallengerAI(initial_difficulty_range=(0.2, 0.6))
    
    # Initialize the PyTorch-based SolverAI
    solver_ai = SolverAI(
        device=device,
        tokenizer=tokenizer,
        emb_size=EMB_SIZE,
        nhead=NHEAD,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        ffn_hid_dim=FFN_HID_DIM,
        dropout=DROPOUT,
        learning_rate=LEARNING_RATE,
        max_seq_len=MAX_SEQ_LEN
    )
    
    evaluator = Evaluator()

    orchestrator = TrainingOrchestrator(challenger_ai, solver_ai, evaluator)
    
    print(f"\nStarting training on device: {device}")
    orchestrator.run_training_loop(num_iterations=NUM_ITERATIONS)

    # You can add code here to save the trained solver model, tokenizer, etc.
    # e.g., torch.save(solver_ai.model.state_dict(), "solver_model.pth")

    print("\n--- Example Post-Training Interaction (Conceptual) ---")
    # This part would use the trained solver_ai.solve() method.
    # For a real test, you might want to generate a few problems without training in between.
    
    # Let's try one problem with the trained solver
    if NUM_ITERATIONS > 0: # Ensure some training happened
        print("\nGenerating one final test problem...")
        test_problem = challenger_ai.create_problem()
        if test_problem:
            # Ensure solver is in eval mode for this test if not already handled by solve()
            solver_ai.model.eval() 
            solution = solver_ai.solve(test_problem) # solver.solve should handle model.eval() internally
            evaluation_results = evaluator.evaluate(test_problem, solution)
            print(f"Test Problem: {test_problem.text}")
            print(f"Solver's Final Solution: {solution}")
            print(f"Evaluation: Correct? {evaluation_results['is_correct']} - Detail: {evaluation_results['detail']}")
            if not evaluation_results['is_correct'] and test_problem.expected_solution_expr:
                expected_sol_str = sympy.printing.sstr(test_problem.expected_solution_expr)
                if isinstance(test_problem.expected_solution_expr, list):
                    expected_sol_str = ", ".join([sympy.printing.sstr(eq) for eq in test_problem.expected_solution_expr])
                print(f"Expected Solution for test: {expected_sol_str}")
    else:
        print("No training iterations were run, skipping post-training test.")