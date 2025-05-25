import numpy as np
import sympy
from .data_models import MathProblem
from .challenger import ChallengerAI
from .solver import SolverAI
from .evaluator import Evaluator

class TrainingOrchestrator:
    def __init__(self, challenger: ChallengerAI, solver: SolverAI, evaluator: Evaluator):
        self.challenger = challenger
        self.solver = solver
        self.evaluator = evaluator
        self.iteration_count = 0
        self.history = []

    def _calculate_solvability_reward(self, problem: MathProblem, evaluation_results: dict) -> float:
        """
        Calculates reward for the Challenger.
        - Positive if solved correctly, scaled by difficulty and effort.
        - Negative if too hard (unsolved) or if problem was trivial.
        """
        is_correct = evaluation_results["is_correct"]
        problem_difficulty = problem.generated_difficulty_score # Challenger's own estimate
        # solver_effort = evaluation_results["solver_effort_proxy"] # Not well implemented yet

        if is_correct:
            # Reward for creating a solvable problem, more if it was appropriately challenging.
            # If solved correctly, give base + bonus for difficulty.
            # Let's say optimal difficulty is around 0.5-0.7 for the solver
            # We want to reward problems that are solved AND were not trivial
            reward = 0.5 + problem_difficulty * 0.5 # Base for solved, bonus for perceived difficulty
            # if solver_effort < 0.2 and problem_difficulty > 0.5: # Solved too easily for its difficulty
            #     reward *= 0.7 # Penalize slightly for being too easy for solver
        else:
            # Penalize for creating an unsolvable problem (for current solver)
            # More penalty if it was rated as "easy" by challenger but still failed (means bad rating or too hard)
            reward = -0.5 - (1.0 - problem_difficulty) * 0.5 # Base penalty, more if it was "easy" but failed
            if "Solver provided an empty solution" in evaluation_results["detail"] or \
               "No valid solutions parsed" in evaluation_results["detail"] or \
               "Error parsing solver output" in evaluation_results["detail"] :
                # Solver couldn't even produce a parsable output
                reward -= 0.2 # Extra penalty if solver was completely stumped or errored badly
        return np.clip(reward, -1.0, 1.0)

    def _calculate_solved_reward(self, problem: MathProblem, evaluation_results: dict) -> float:
        """
        Calculates reward for the Solver.
        - Positive if solved correctly, scaled by problem difficulty.
        - Negative if incorrect.
        """
        is_correct = evaluation_results["is_correct"]
        problem_difficulty = problem.generated_difficulty_score

        if is_correct:
            # Reward for solving, more if problem was harder.
            reward = 0.5 + problem_difficulty * 1.0 # Bonus for solving harder problems
        else:
            reward = -1.0 # Flat penalty for being wrong
        return np.clip(reward, -1.0, 1.0)


    def run_training_iteration(self):
        self.iteration_count += 1
        print(f"\n--- Iteration {self.iteration_count} ---")

        # 1. Challenger creates a problem
        problem = self.challenger.create_problem()
        if not problem:
            print("Orchestrator: Challenger failed to create a problem. Skipping.")
            return

        # 2. Solver attempts to solve the problem
        solution_str = self.solver.solve(problem)

        # 3. Evaluator assesses the Problem and Solution
        evaluation_results = self.evaluator.evaluate(problem, solution_str)
        print(f"Evaluator: Assessment for Problem ID {problem.id} - Correct: {evaluation_results['is_correct']}. Detail: {evaluation_results['detail']}")
        if not evaluation_results['is_correct'] and problem.expected_solution_expr:
             expected_sol_str = "N/A"
             if isinstance(problem.expected_solution_expr, list):
                expected_sol_str = ", ".join([sympy.printing.pretty(eq) for eq in problem.expected_solution_expr])
             else:
                expected_sol_str = sympy.printing.pretty(problem.expected_solution_expr)
             print(f"Evaluator: Expected solution was '{expected_sol_str}'")


        # 4. Calculate Rewards based on Evaluation
        solvability_reward = self._calculate_solvability_reward(problem, evaluation_results)
        solved_reward = self._calculate_solved_reward(problem, evaluation_results)

        # 5. Challenger receives feedback and reward
        self.challenger.update(problem, evaluation_results, solvability_reward)

        # 6. Solver receives feedback and reward
        self.solver.update(problem, solution_str, evaluation_results, solved_reward)
        
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
            "solver_confidence_after": self.solver.confidence,
            "challenger_type_weights_after": {k: v["weight"] for k,v in self.challenger.problem_type_focus.items()}
        })


    def run_training_loop(self, num_iterations=50):
        print("--- Starting DML Training Loop (Diagram Aligned) ---")
        for _ in range(num_iterations):
            self.run_training_iteration()
            if self.iteration_count % 10 == 0:
                print("\n*** Mid-loop summary (last 10 iterations) ***")
                last_10 = self.history[-10:]
                if last_10:
                    avg_solvability_R = np.mean([h['solvability_reward'] for h in last_10])
                    avg_solved_R = np.mean([h['solved_reward'] for h in last_10])
                    correct_count = sum(1 for h in last_10 if h['evaluation']['is_correct'])
                    print(f"Avg Solvability Reward (Challenger): {avg_solvability_R:.2f}")
                    print(f"Avg Solved Reward (Solver): {avg_solved_R:.2f}")
                    print(f"Solver Accuracy: {correct_count}/10")
                    print(f"Solver Confidence: {self.solver.confidence:.2f}")
                    print(f"Challenger Type Weights: {self.history[-1]['challenger_type_weights_after']}")


        print("\n--- DML Training Loop Finished ---")