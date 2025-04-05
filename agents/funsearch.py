from agents.utils import call_llm, extract_code_blocks, textwrap
from agents.funsearch_agent.model import Funsearch as FunsearchAgent
from dataclasses import dataclass
from typing import Optional


@dataclass
class Solution:
    prompt_id: int
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None


class FunSearch:
    def __init__(self,
                 problem_description,
                 timeout=10,
                 model='openai/o3-mini',
                 max_iter=64,
                 reasoning_effort='medium',
                 num_islands=10,
                 functions_per_prompt=2,
                 reset_period=2 * 60 * 60
                 ):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort

        self.agent = FunsearchAgent(problem_description,
                                    num_islands=num_islands,
                                    functions_per_prompt=functions_per_prompt,
                                    reset_period=reset_period,
                                    timeout=timeout)

    def step(self):
        prompt, prompt_id = self.agent.get_prompt()
        response = call_llm(prompt, model=self.model, reasoning_effort=self.reasoning_effort)
        code_blocks = extract_code_blocks(response)
        code = textwrap.dedent(code_blocks[0])
        self.solution.append(Solution(prompt_id=prompt_id, code=code, response=response))
        return code

    def feedback(self, score, feedback):
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback
        self.agent.pull_score(self.solution[-1].prompt_id, score, feedback, self.solution[-1].code)
        return

    def finalize(self):
        previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
        return previous_best.code

