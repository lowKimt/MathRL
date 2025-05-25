import random
import numpy as np
import sympy
from sympy import solve, symbols, Eq, S # S is the SymPy singleton registry for True/False
from .data_models import MathProblem # Import MathProblem from the new data_models.py

class ChallengerAI:
    def __init__(self, initial_difficulty_range=(0.1, 0.3)):
        self.problem_id_counter = 0
        self.difficulty_target = sum(initial_difficulty_range) / 2
        self.difficulty_range = list(initial_difficulty_range)
        self.problem_type_focus = {
            "linear_equation_1var": {"weight": 1.0, "params": {"coeff_range": (1,5), "const_range": (0,10)}, "success_rate": 0.5, "attempts":0},
            "quadratic_equation_1var": {"weight": 0.5, "params": {"coeff_range": (1,3), "const_range": (0,5)}, "success_rate": 0.5, "attempts":0},
        }
        # (Code for _generate_linear_equation_1var and _generate_quadratic_equation_1var from previous example
        # can be copied here, they are good. Ensure they set problem.equation_form_sympy)
    def _generate_linear_equation_1var(self, params):
        x = symbols('x')
        a = random.randint(*params["coeff_range"])
        while a == 0: a = random.randint(*params["coeff_range"])
        b = random.randint(*params["const_range"])
        sol_val = random.randint(-10, 10)
        c_val = a * sol_val + b
        eq_lhs = a*x + b
        eq_rhs = sympy.Integer(c_val)
        problem_statement_sympy = Eq(eq_lhs, eq_rhs)
        problem_text = f"Solve for x: {sympy.printing.pretty(problem_statement_sympy)}"
        expected_solution_expr = Eq(x, sympy.Integer(sol_val))
        difficulty = np.clip((abs(a) + abs(b) + abs(c_val)) / 50.0, 0.05, 0.95)
        return problem_text, problem_statement_sympy, [x], expected_solution_expr, difficulty

    def _generate_quadratic_equation_1var(self, params):
        x = symbols('x')
        a = random.randint(*params["coeff_range"])
        while a == 0: a = random.randint(*params["coeff_range"])
        r1 = random.randint(-5, 5)
        r2 = random.randint(-5, 5)
        # if r1 == r2: r2 +=1 # ensure distinct roots for more interesting quadratics sometimes
        b = -a * (r1 + r2)
        c = a * r1 * r2
        problem_statement_sympy = Eq(a*x**2 + b*x + c, 0)
        problem_text = f"Solve for x: {sympy.printing.pretty(problem_statement_sympy)}"
        sols = solve(problem_statement_sympy, x)
        if not sols: return None
        expected_solution_expr = sorted([Eq(x, sol) for sol in sols], key=str) # Canonical form
        difficulty = np.clip((abs(a) + abs(b) + abs(c)) / 100.0, 0.1, 0.95)
        return problem_text, problem_statement_sympy, [x], expected_solution_expr, difficulty


    def create_problem(self) -> MathProblem | None:
        self.problem_id_counter += 1
        types = list(self.problem_type_focus.keys())
        raw_weights = np.array([self.problem_type_focus[t]["weight"] for t in types])
        # Add small epsilon to weights to avoid all zeros if weights become too small
        probabilities = (raw_weights + 1e-6) / (raw_weights.sum() + 1e-6 * len(raw_weights))

        chosen_type = random.choices(types, weights=probabilities, k=1)[0]
        params = self.problem_type_focus[chosen_type]["params"]
        gen_output = None

        if chosen_type == "linear_equation_1var":
            gen_output = self._generate_linear_equation_1var(params)
        elif chosen_type == "quadratic_equation_1var":
            gen_output = self._generate_quadratic_equation_1var(params)

        if gen_output is None: return None
        problem_text, problem_eq_sympy, variables, expected_solution, difficulty = gen_output

        problem = MathProblem(self.problem_id_counter, problem_text, chosen_type, variables, expected_solution, difficulty)
        problem.equation_form_sympy = problem_eq_sympy
        print(f"Challenger: Created Problem ID {problem.id} (Est. Diff: {problem.generated_difficulty_score:.2f}) - {problem.text}")
        return problem

    def update(self, problem:MathProblem, evaluation_results: dict, solvability_reward: float):
        """ Challenger learns from the solvability_reward and evaluation_results. """
        print(f"Challenger: Received Solvability Reward: {solvability_reward:.2f} for Problem ID {problem.id}. Eval Detail: '{evaluation_results['detail']}'")
        
        # Update success rate for the problem type
        ptype_stats = self.problem_type_focus[problem.type]
        ptype_stats["attempts"] = ptype_stats.get("attempts",0) + 1 # Ensure attempts is initialized
        if evaluation_results["is_correct"]:
            ptype_stats["success_rate"] = (ptype_stats["success_rate"] * (ptype_stats["attempts"]-1) + 1) / ptype_stats["attempts"]
        else:
            ptype_stats["success_rate"] = (ptype_stats["success_rate"] * (ptype_stats["attempts"]-1)) / ptype_stats["attempts"]


        # APS-like adaptation:
        # If reward is high (problem was good), slightly increase weight for this type
        if solvability_reward > 0.5: # Good problem
            ptype_stats["weight"] = min(5.0, ptype_stats["weight"] * 1.1)
            # If this type is consistently solved, try making it harder
            if ptype_stats["success_rate"] > 0.7:
                 current_params = ptype_stats["params"]
                 current_params["coeff_range"] = (current_params["coeff_range"][0], current_params["coeff_range"][1] + 1)
                 current_params["const_range"] = (current_params["const_range"][0], current_params["const_range"][1] + 1)


        # If reward is low (problem too hard or too easy in a bad way)
        elif solvability_reward < -0.2: # Problem was too hard or solver failed badly
            ptype_stats["weight"] = max(0.1, ptype_stats["weight"] * 0.9) # Reduce focus
            # If this type is consistently failed, try making it easier
            if ptype_stats["success_rate"] < 0.3:
                current_params = ptype_stats["params"]
                current_params["coeff_range"] = (max(1, current_params["coeff_range"][0]-1), max(2,current_params["coeff_range"][1]-1))


        # Normalize weights so they don't grow indefinitely and sum to something reasonable (e.g. total_types)
        total_weight = sum(self.problem_type_focus[t]["weight"] for t in self.problem_type_focus)
        num_types = len(self.problem_type_focus)
        for t in self.problem_type_focus:
            self.problem_type_focus[t]["weight"] = (self.problem_type_focus[t]["weight"] / total_weight) * num_types


        # Adjust overall difficulty target based on average solver performance (more implicit here)
        # The diagram implies evaluator output feeds reward, reward then updates model.
        # So this direct update is driven by the scalar reward.