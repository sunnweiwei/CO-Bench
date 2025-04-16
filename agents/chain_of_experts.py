from agents.chain_expert.utils import extract_code_from_string
from agents.chain_expert.main import chain_of_experts
from agents.chain_expert.test_generated_code import test_generated_code
from agents.chain_expert.result import Result
from agents.chain_expert.evaluator import Evaluator



def solve_with_chain_of_experts(problem_description, solve_template,
                                model_name='gpt-3.5-turbo',
                                max_collaborate_nums=5,
                                enable_reflection=True,
                                max_trials=8):
    # Prepare the problem data structure expected by chain_of_experts
    problem_data = {
        'description': problem_description,
        'code_example': solve_template
    }

    # Call the chain_of_experts function from the original implementation
    answer = chain_of_experts(
        problem=problem_data,
        max_collaborate_nums=max_collaborate_nums,
        model_name=model_name,
        enable_reflection=enable_reflection,
        max_trials=max_trials
    )

    # Extract the code from the answer
    code = extract_code_from_string(answer)

    return code, answer

def split_solve_parts(text: str) -> tuple:
    """Split input text into two parts: before 'def solve' and from 'def solve' onward."""
    idx = text.find("def solve")
    return (text[:idx].rstrip(), text[idx:].lstrip()) if idx != -1 else (text, "")


class ChainOfExperts:
    def __str__(self):
        return f"Chain Of Experts"

    def __init__(self, problem_description, timeout=10, model='openai/o3-mini',
                 reasoning_effort='medium', max_collaborate_nums=5, enable_reflection=True, max_trials=8):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = None
        self.answer = None
        self.reasoning_effort = reasoning_effort
        self.max_collaborate_nums = max_collaborate_nums
        self.enable_reflection = enable_reflection
        self.max_trials = max_trials

    def step(self):
        if self.solution is not None:
            return None

        description, solve_template = split_solve_parts(self.problem_description)
        code, answer = solve_with_chain_of_experts(description, solve_template,
                                    model_name=self.model,
                                    max_collaborate_nums=self.max_collaborate_nums,
                                    enable_reflection=self.enable_reflection,
                                    max_trials=self.max_trials)
        self.answer = answer
        self.solution = code
        return code

    def feedback(self, score, feedback, ):
        return

    def finalize(self):
        return self.solution



