import sympy
from sympy.parsing.sympy_parser import parse_expr
from sympy import Eq, S # S is the SymPy singleton registry for True/False
from .data_models import MathProblem # Import MathProblem from the new data_models.py

class Evaluator:
    def __init__(self):
        pass # Can have configuration for evaluation strictness, etc.

    def evaluate(self, problem: MathProblem, solver_solution_str: str) -> dict:
        """
        Evaluates the solver's solution for a given problem.
        Returns a dictionary with evaluation results.
        """
        is_correct = False
        parsed_solver_expr = None
        detail_message = ""
        solver_effort_proxy = 0.5 # Placeholder, ideally solver would report this or it's timed

        try:
            # Attempt to parse the solver's output string
            parsed_solver_solutions = []
            assignments = solver_solution_str.split(',')
            if not solver_solution_str.strip(): # Handle empty solution string
                detail_message = "Solver provided an empty solution."
            else:
                for assignment in assignments:
                    assignment_strip = assignment.strip()
                    if not assignment_strip: continue

                    if '=' not in assignment_strip:
                        if len(problem.variables) == 1 and not problem.expected_solution_expr.free_symbols:
                            parsed_solver_solutions.append(parse_expr(assignment_strip))
                        else:
                            detail_message = f"Invalid format for assignment '{assignment_strip}', expected 'var = value'."
                            break
                        continue

                    var_name, value_str = assignment_strip.split('=', 1)
                    var_name = var_name.strip()
                    value_str = value_str.strip()

                    target_var = next((v for v in problem.variables if str(v) == var_name), None)
                    if not target_var:
                        detail_message = f"Variable '{var_name}' in solution is not in problem variables {problem.variables}."
                        break
                    parsed_solver_solutions.append(Eq(target_var, parse_expr(value_str)))
                else: # If loop completed without break
                    if not parsed_solver_solutions and solver_solution_str.strip():
                         detail_message = "No valid solutions parsed from non-empty solver output."


            if not detail_message and not parsed_solver_solutions: # No solutions parsed and no specific error yet
                if solver_solution_str.strip(): # If solver provided something but it wasn't parsable to equations
                    detail_message = "Solver output was non-empty but could not be parsed into equation(s)."
                # If solver_solution_str was empty, detail_message would be set already.

            # Proceed with verification if parsing was okay (or if we check against unparsed expected solution)
            if not detail_message:
                if problem.expected_solution_expr is not None:
                    # Using sympy's .equals() for comparison of expressions/equations
                    expected_sols = problem.expected_solution_expr
                    if not isinstance(expected_sols, list): expected_sols = [expected_sols]
                    
                    # Sort both for canonical comparison (important for sets of solutions)
                    parsed_solver_solutions_sorted = sorted(parsed_solver_solutions, key=lambda x: str(x))
                    expected_sols_sorted = sorted(expected_sols, key=lambda x: str(x))

                    if len(parsed_solver_solutions_sorted) == len(expected_sols_sorted):
                        match = all(s_eq.equals(e_eq) for s_eq, e_eq in zip(parsed_solver_solutions_sorted, expected_sols_sorted))
                        if match:
                            is_correct = True
                            detail_message = "Solution is correct (matches expected)."
                            parsed_solver_expr = parsed_solver_solutions # Store list of Eq
                        else:
                            detail_message = "Solution does not match expected solution."
                    else:
                        detail_message = f"Incorrect number of solutions. Got {len(parsed_solver_solutions_sorted)}, expected {len(expected_sols_sorted)}."
                    
                    if parsed_solver_expr is None and parsed_solver_solutions: # store first parsed if not correct but parsable
                         parsed_solver_expr = parsed_solver_solutions[0] if len(parsed_solver_solutions) == 1 else parsed_solver_solutions


                elif hasattr(problem, 'equation_form_sympy') and problem.equation_form_sympy:
                    # Verification by substitution if no ground truth from Challenger (more complex)
                    problem_eqs = problem.equation_form_sympy
                    if not isinstance(problem_eqs, list): problem_eqs = [problem_eqs]

                    subs_dict = {}
                    for sol_eq in parsed_solver_solutions:
                        if isinstance(sol_eq, sympy.Equality): # e.g. Eq(x, 5)
                            subs_dict[sol_eq.lhs] = sol_eq.rhs
                        # else: could be a direct expression, e.g. if problem was "evaluate 2+2"
                        # and solver returned "4". This case needs specific handling.

                    all_hold = True
                    if not subs_dict and parsed_solver_solutions: # if solutions were expressions, not equations
                        # This path is less common for "solve for x" type problems
                        # For "2+2", expected_solution_expr should be `Integer(4)`
                        # If problem is `Eq(x, 2+2)` and solver gives `x=4`, subs_dict works.
                        pass # Let it proceed, might fail if problem_eqs need substitutions.
                    
                    if not parsed_solver_solutions and not solver_solution_str.strip(): # Empty solution
                         all_hold = False # Cannot satisfy equations with empty solution typically.
                         detail_message = "Empty solution cannot satisfy the problem equations."

                    if parsed_solver_solutions: # only try substitution if there's something to substitute
                        for p_eq in problem_eqs:
                            try:
                                # .subs().simplify() should yield True (or S.true) if eq holds
                                if p_eq.subs(subs_dict).simplify() is not S.true:
                                    all_hold = False
                                    break
                            except Exception as e_subs:
                                all_hold = False
                                detail_message = f"Error during substitution: {e_subs}"
                                break
                        if all_hold:
                            is_correct = True
                            detail_message = "Solution is correct (verified by substitution)."
                            parsed_solver_expr = parsed_solver_solutions
                        elif not detail_message: # if all_hold is false and no specific error
                            detail_message = "Solution is incorrect (failed substitution)."
                else:
                    detail_message = "Cannot verify: No expected solution or parsable problem equation provided."

        except (SyntaxError, TypeError, sympy.SympifyError) as e:
            detail_message = f"Error parsing solver output '{solver_solution_str}': {e}"
        except Exception as e_gen: # Catch any other unexpected errors during evaluation
            detail_message = f"Unexpected error during evaluation: {e_gen}"


        if not detail_message: # Default message if nothing else set it
            detail_message = "Evaluation inconclusive."
        if is_correct and parsed_solver_expr is None and parsed_solver_solutions: # ensure parsed_solver_expr is set if correct
            parsed_solver_expr = parsed_solver_solutions[0] if len(parsed_solver_solutions) == 1 else parsed_solver_solutions


        return {
            "is_correct": is_correct,
            "parsed_solver_expr": parsed_solver_expr, # Could be single Eq, list of Eq, or other expr
            "solver_effort_proxy": solver_effort_proxy, # TODO: Integrate this better
            "problem_difficulty_estimate": problem.generated_difficulty_score, # From Challenger
            "detail": detail_message
        }