from agents.utils import call_llm, extract_code_blocks, textwrap
from dataclasses import dataclass
from typing import Optional

@dataclass
class Solution:
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None


class GreedyRefine:
    def __str__(self):
        return f"Greedy Refinement"

    def __init__(self, problem_description, timeout=10, model='openai/o3-mini', max_iter=64, reasoning_effort='medium'):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort

    def step(self):
        if len(self.solution) == 0:
            prompt = (
                f"You are an expert in Operation Research problem. "
                f"Solve the following problem:\n\n{self.problem_description}\n\n"
                f"Ensure your algorithm is as effective as possible. You may use any Python package. "
                f"Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs)`. "
                f"Your function has a {self.timeout}-second timeout; aim to return the best possible results within this limit."
            )
        else:
            previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
            prompt = (
                f"You are an expert in Operations Research."
                f" You are tasked with solving the following problem:\n\n{self.problem_description}\n\n"
                f"Below is a previously developed solution. Your goal is to enhance this solution to further improve its test-time performance:\n\n"
                f"{previous_best.code}\n\n"
                f"Here are the evaluation scores of the existing solution for each test case and example:\n\n"
                f"{previous_best.feedback}\n\n"
                f"These scores are normalized relative to a reference solution, with higher values indicating better performance. "
                f"Analyze these evaluation results carefully to identify areas for improvement.\n\n"
                f"First, outline a concise, clear plan in natural language describing how you intend to improve the solution."
                f" Then, implement your proposed improvements in Python based on the previous solution provided. "
                f"You are encouraged to propose significant, innovative improvements—your solution should be distinctly different and clearly superior. "
                f"If you have a completely new and more effective approach, feel free to abandon the previous method and adopt your new approach. "
                f"Enclose all your code within a Python code block using: ```python ... ``` and ensure the main function is named `def solve(**kwargs)`. "
                f"Do not use separator lines (e.g., '-----'). "
                f"Ensure your code is as effective as possible. You may use any Python package. "
                f"Your function has a {self.timeout}-second timeout; aim to return the best possible results within this limit."
            )
        response = call_llm(prompt, model=self.model, reasoning_effort=self.reasoning_effort)
        code_blocks = extract_code_blocks(response)
        code = textwrap.dedent(code_blocks[0])
        self.solution.append(Solution(code=code, response=response))
        return code

    def feedback(self, score, feedback):
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback
        return

    def finalize(self):
        previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
        return previous_best.code