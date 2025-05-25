import sympy

class MathProblem:
    def __init__(self, problem_id: int, text_representation: str,
                 problem_type: str, variables: list,
                 expected_solution_expr: sympy.Expr = None, # Ground truth sympy expression
                 generated_difficulty_score: float = 0.5): # Challenger's estimate
        self.id = problem_id
        self.text = text_representation
        self.type = problem_type
        self.variables = variables
        self.expected_solution_expr = expected_solution_expr
        self.generated_difficulty_score = generated_difficulty_score
        self.equation_form_sympy = None # Will be set by Challenger

    def __repr__(self):
        return f"Problem(id={self.id}, type='{self.type}', text='{self.text}')"