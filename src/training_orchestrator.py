# src/training_orchestrator.py
import numpy as np
import sympy
from .data_models import MathProblem
from .challenger import ChallengerAI
from .solver import SolverAI # Will be the new PyTorch based SolverAI
from .evaluator import Evaluator
# No PyTorch specific imports needed here directly unless we pass device through it

class TrainingOrchestrator:
    # __init__ will now expect solver to be the PyTorch SolverAI instance
    def __init__(self, challenger: ChallengerAI, solver: SolverAI, evaluator: Evaluator):
        self.challenger = challenger
        self.solver = solver # This is now the PyTorch SolverAI
        self.evaluator = evaluator
        self.iteration_count = 0
        self.history = []

    # _calculate_solvability_reward and _calculate_solved_reward remain the same conceptually
    def _calculate_solvability_reward(self, problem: MathProblem, evaluation_results: dict) -> float:
        is_correct = evaluation_results["is_correct"]
        problem_difficulty = problem.generated_difficulty_score
        reward = 0.0
        if is_correct:
            reward = 0.5 + problem_difficulty * 0.5 
        else:
            reward = -0.5 - (1.0 - problem_difficulty) * 0.5
            if "Solver provided an empty solution" in evaluation_results["detail"] or \
               "No valid solutions parsed" in evaluation_results["detail"] or \
               "Error parsing solver output" in evaluation_results["detail"] :
                reward -= 0.2 
        return np.clip(reward, -1.0, 1.0)

    def _calculate_solved_reward(self, problem: MathProblem, evaluation_results: dict) -> float:
        is_correct = evaluation_results["is_correct"]
        problem_difficulty = problem.generated_difficulty_score
        reward = 0.0
        if is_correct:
            reward = 0.5 + problem_difficulty * 1.0
        else:
            reward = -1.0 
        return np.clip(reward, -1.0, 1.0)

    def run_training_iteration(self): # Content mostly same, just uses the new Solver
        self.iteration_count += 1
        print(f"\n--- Iteration {self.iteration_count} ---")

        problem = self.challenger.create_problem()
        if not problem:
            print("Orchestrator: Challenger failed to create a problem. Skipping.")
            return

        # Solver now uses its neural model for solve() and learns in update()
        solution_str = self.solver.solve(problem) # Uses PyTorch model for inference

        evaluation_results = self.evaluator.evaluate(problem, solution_str)
        print(f"Evaluator: Assessment for Problem ID {problem.id} - Correct: {evaluation_results['is_correct']}. Detail: {evaluation_results['detail']}")
        if not evaluation_results['is_correct'] and problem.expected_solution_expr:
             expected_sol_str = "N/A"
             if isinstance(problem.expected_solution_expr, list):
                expected_sol_str = ", ".join([sympy.printing.sstr(eq) for eq in problem.expected_solution_expr])
             else:
                expected_sol_str = sympy.printing.sstr(problem.expected_solution_expr)
             print(f"Evaluator: Expected solution was '{expected_sol_str}'")

        solvability_reward = self._calculate_solvability_reward(problem, evaluation_results)
        solved_reward = self._calculate_solved_reward(problem, evaluation_results)

        self.challenger.update(problem, evaluation_results, solvability_reward)
        
        # Solver's update method now performs a training step
        self.solver.update(problem, solution_str, evaluation_results, solved_reward) 
        
        current_solver_loss = self.solver.train_history["loss"][-1] if self.solver.train_history["loss"] else float('nan')
        
        self.history.append({
            "iteration": self.iteration_count,
            "problem_id": problem.id,
            "problem_text": problem.text,
            "problem_type": problem.type,
            "challenger_difficulty_est": problem.generated_difficulty_score,
            "solver_solution": solution_str,
            "evaluation": evaluation_results,
            "solvability_reward": solvability_reward,
            "solved_reward": solved_reward,
            # "solver_confidence_after": self.solver.confidence, # Old solver attribute
            "solver_loss_after": current_solver_loss, # New attribute
            "challenger_type_weights_after": {k: v["weight"] for k,v in self.challenger.problem_type_focus.items()}
        })

    def run_training_loop(self, num_iterations=50): # Content mostly same
        print("--- Starting DML Training Loop (PyTorch Solver) ---")
        for _ in range(num_iterations):
            self.run_training_iteration()
            if self.iteration_count > 0 and self.iteration_count % 10 == 0 : # Check history has items
                print("\n*** Mid-loop summary (last 10 iterations) ***")
                last_10 = self.history[-10:]
                if last_10: # Ensure last_10 is not empty
                    avg_solvability_R = np.mean([h['solvability_reward'] for h in last_10])
                    avg_solved_R = np.mean([h['solved_reward'] for h in last_10])
                    correct_count = sum(1 for h in last_10 if h['evaluation']['is_correct'])
                    avg_loss = np.mean([h['solver_loss_after'] for h in last_10 if not np.isnan(h['solver_loss_after'])])
                    
                    print(f"Avg Solvability Reward (Challenger): {avg_solvability_R:.2f}")
                    print(f"Avg Solved Reward (Solver): {avg_solved_R:.2f}")
                    print(f"Solver Accuracy (last 10): {correct_count}/10")
                    if not np.isnan(avg_loss):
                         print(f"Avg Solver Loss (last 10 with training): {avg_loss:.4f}")
                    else:
                         print(f"Avg Solver Loss (last 10 with training): No training steps recorded or all NaN.")

                    if self.history and "challenger_type_weights_after" in self.history[-1]: # check if key exists
                        print(f"Challenger Type Weights: {self.history[-1]['challenger_type_weights_after']}")
        print("\n--- DML Training Loop Finished ---")