from agents.utils import call_llm, extract_code_blocks, textwrap
from dataclasses import dataclass
from typing import Optional
import queue


@dataclass
class Solution:
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None


class BestOfN:
    def __str__(self):
        return f"Greedy Refinement"

    def __init__(self, problem_description, timeout=10, model='openai/o3-mini', max_iter=64,
                 reasoning_effort='medium'):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort

    def step(self):
        prompt = (
                f"You are an expert in Operation Research problem. "
                f"Solve the following problem:\n\n{self.problem_description}\n\n"
                f"Ensure your algorithm is as effective as possible. You may use any Python package. "
                f"Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs)`. "
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
