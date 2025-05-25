import random
import sympy
from sympy import solve, S # S is the SymPy singleton registry for True/False
from .data_models import MathProblem # Import MathProblem from the new data_models.py

class SolverAI:
    def __init__(self):
        self.solver_engine = sympy.solve # Basic symbolic solver
        self.confidence = 1.0 # A mock confidence level
        self.learning_rate = 0.05 # For mock confidence update

    def solve(self, problem: MathProblem) -> str:
        print(f"Solver: Attempting Problem ID {problem.id}: {problem.text}")
        try:
            if hasattr(problem, 'equation_form_sympy') and problem.equation_form_sympy:
                solutions = self.solver_engine(problem.equation_form_sympy, problem.variables if len(problem.variables)>1 else problem.variables[0])
            else: # Fallback (less reliable)
                # This part needs robust parsing of problem.text if equation_form_sympy isn't available
                # For simplicity, we assume Challenger always provides equation_form_sympy for this Solver.
                raise ValueError("Solver requires problem.equation_form_sympy for reliable solving.")

            solution_str = ""
            if isinstance(solutions, dict):
                solution_str = ", ".join([f"{str(k)} = {str(v)}" for k, v in sorted(solutions.items(), key=lambda item: str(item[0]))])
            elif isinstance(solutions, list):
                if not solutions: solution_str = "" # Represent "no solution found" as empty
                elif isinstance(solutions[0], tuple): # System solutions
                    solution_str = ", ".join([f"{str(problem.variables[i])} = {str(s)}" for i, s in enumerate(solutions[0])])
                else: # Multiple solutions for a single variable
                    var_name = str(problem.variables[0])
                    solution_str = ", ".join([f"{var_name} = {str(s)}" for s in sorted(solutions, key=str)])
            elif solutions is S.true or solutions is S.false: # e.g. for identities or contradictions
                 solution_str = str(solutions)
            else: # Single solution value
                solution_str = f"{str(problem.variables[0])} = {str(solutions)}"
            
            # Simulate slight variation/error sometimes if confidence is low (for training)
            if self.confidence < 0.6 and random.random() < 0.2: # 20% chance of small error if not confident
                if solution_str and "=" in solution_str:
                    parts = solution_str.split("=")
                    try:
                        val = float(parts[-1].strip())
                        val += random.choice([-1,1]) * random.random() # Add small noise
                        solution_str = f"{parts[0].strip()} = {val:.2f}"
                        print(f"Solver: (Low confidence, introduced noise) -> {solution_str}")
                    except ValueError: pass # Not a float, can't easily add noise

            print(f"Solver: Proposed solution for Problem ID {problem.id}: '{solution_str}'")
            return solution_str

        except Exception as e:
            print(f"Solver: Error solving Problem ID {problem.id}: {e}")
            return "" # Return empty string on error, Evaluator will handle it

    def update(self, problem: MathProblem, proposed_solution_str:str, evaluation_results: dict, solved_reward: float):
        """ Solver learns from the solved_reward and evaluation_results. """
        print(f"Solver: Received Solved Reward: {solved_reward:.2f} for Problem ID {problem.id}. Eval Detail: '{evaluation_results['detail']}'")
        # For a real ML solver, this is where backprop/model update happens.
        # For this mock solver, let's adjust "confidence".
        if evaluation_results["is_correct"]:
            self.confidence = min(1.0, self.confidence + self.learning_rate * solved_reward)
        else:
            # Penalize more heavily if reward is very negative
            self.confidence = max(0.1, self.confidence + self.learning_rate * solved_reward) 
        print(f"Solver: New confidence level: {self.confidence:.2f}")