from agents.utils import call_llm, extract_code_blocks, textwrap
import agents.aide_agent as aide
from agents.aide_agent.utils.config import save_run
from agents.aide_agent.interpreter import ExecutionResult
from agents.aide_agent.journal import Node
from dataclasses import dataclass
from typing import Optional
import os


@dataclass
class Solution:
    plan: str
    code: str
    result_node: Node
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None


class AIDE:
    def __init__(self,
                 problem_description,
                 timeout=10,
                 model='openai/o3-mini',
                 max_iter=64,
                 reasoning_effort='medium',
                 ):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.max_iter = max_iter
        self.reasoning_effort = reasoning_effort
        os.makedirs(os.path.join(os.getcwd(), 'Tmp'), exist_ok=True)
        self.exp = aide.Experiment(
            data_dir=os.path.join(os.getcwd(), 'Tmp'),  # replace this with your own directory
            goal=problem_description,
        )
        self.exp.agent.acfg.code.model = model
        self.exp.agent.acfg.feedback.model = 'gpt-4o-mini'
        self.exp.agent.cfg.exec.timeout = timeout

    def step(self):
        if not self.exp.agent.journal.nodes or self.exp.agent.data_preview is None:
            self.exp.agent.update_data_preview()
        parent_node = self.exp.agent.search_policy()

        if parent_node is None:
            result_node = self.exp.agent.draft()
        elif parent_node.is_buggy:
            result_node = self.exp.agent.debug(parent_node)
        else:
            result_node = self.exp.agent.improve(parent_node)

        code = result_node.code
        plan = result_node.plan
        self.solution.append(Solution(plan=plan, code=code, result_node=result_node))
        return code

    def feedback(self, score, feedback):
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback

        exec_result = ExecutionResult(feedback.split('\n'), self.timeout, None)
        self.exp.agent.parse_exec_result(node=self.solution[-1].result_node, exec_result=exec_result)
        self.exp.agent.journal.append(self.solution[-1].result_node)
        return

    def finalize(self):
        previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
        return previous_best.code

