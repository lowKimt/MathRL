from src.challenger import ChallengerAI
from src.solver import SolverAI
from src.evaluator import Evaluator
from src.training_orchestrator import TrainingOrchestrator

if __name__ == "__main__":
    challenger_ai = ChallengerAI(initial_difficulty_range=(0.2, 0.6))
    solver_ai = SolverAI() # Basic symbolic solver with mock confidence
    evaluator = Evaluator()

    orchestrator = TrainingOrchestrator(challenger_ai, solver_ai, evaluator)
    orchestrator.run_training_loop(num_iterations=30)

    # Print some history for inspection
    # print("\nFull Training History (sample):")
    # for i, entry in enumerate(orchestrator.history[-5:]): # Last 5 entries
    #     print(f"\nEntry {orchestrator.iteration_count - 4 + i}:")
    #     print(f"  Problem ({entry['problem_id']}): {entry['problem_text']} (Type: {entry['problem_type']}, EstDiff: {entry['challenger_difficulty_est']:.2f})")
    #     print(f"  Solver Output: '{entry['solver_solution']}'")
    #     print(f"  Evaluation: Correct? {entry['evaluation']['is_correct']} - {entry['evaluation']['detail']}")
    #     print(f"  Challenger Reward: {entry['solvability_reward']:.2f}")
    #     print(f"  Solver Reward: {entry['solved_reward']:.2f}")
    #     print(f"  Solver Confidence: {entry['solver_confidence_after']:.2f}")
