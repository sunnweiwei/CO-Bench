import logging
import random
from typing import Any, Callable, cast

import humanize
from .backend import FunctionSpec, query
from .interpreter import ExecutionResult
from .journal import Journal, Node
from .utils import data_preview
from .utils.config import Config
from .utils.metric import MetricValue, WorstMetricValue
from .utils.response import extract_code, extract_text_up_to_code, wrap_code

logger = logging.getLogger("aide")

ExecCallbackType = Callable[[str, bool], ExecutionResult]

review_func_spec = FunctionSpec(
    name="submit_review",
    json_schema={
        "type": "object",
        "properties": {
            "is_bug": {
                "type": "boolean",
                "description": "true if the output log shows that the execution failed or has some bug, otherwise false.",
            },
            "summary": {
                "type": "string",
                "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings.",
            },
            "metric": {
                "type": "number",
                "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null.",
            },
            "lower_is_better": {
                "type": "boolean",
                "description": "true if the metric should be minimized (i.e. a lower metric value is better, such as with MSE), false if the metric should be maximized (i.e. a higher metric value is better, such as with accuracy).",
            },
        },
        "required": ["is_bug", "summary", "metric", "lower_is_better"],
    },
    description="Submit a review evaluating the output of the training script.",
)


class Agent:
    def __init__(
            self,
            task_desc: str,
            cfg: Config,
            journal: Journal,
    ):
        super().__init__()
        self.task_desc = task_desc
        self.cfg = cfg
        self.acfg = cfg.agent
        self.journal = journal
        self.data_preview: str | None = None

    def search_policy(self) -> Node | None:
        """Select a node to work on (or None to draft a new node)."""
        search_cfg = self.acfg.search

        # initial drafting
        if len(self.journal.draft_nodes) < search_cfg.num_drafts:
            logger.debug("[search policy] drafting new node (not enough drafts)")
            return None

        # debugging
        if random.random() < search_cfg.debug_prob:
            # nodes that are buggy + leaf nodes + debug depth < max debug depth
            debuggable_nodes = [
                n
                for n in self.journal.buggy_nodes
                if (n.is_leaf and n.debug_depth <= search_cfg.max_debug_depth)
            ]
            if debuggable_nodes:
                logger.debug("[search policy] debugging")
                return random.choice(debuggable_nodes)
            logger.debug("[search policy] not debugging by chance")

        # back to drafting if no nodes to improve
        good_nodes = self.journal.good_nodes
        if not good_nodes:
            logger.debug("[search policy] drafting new node (no good nodes)")
            return None

        # greedy
        greedy_node = self.journal.get_best_node()
        logger.debug("[search policy] greedy node selected")
        return greedy_node

    @property
    def _prompt_environment(self):
        pkgs = [
            "numpy",
            "pandas",
            "networkx",
            "statsmodels",
            "ortools",
            "pulp",
            "pyomo",
            "cvxpy",
            "scipy.sparse.csgraph",
            "deap",
            "gurobipy",
        ]
        random.shuffle(pkgs)
        pkg_str = ", ".join([f"`{p}`" for p in pkgs])

        env_prompt = {
            "Installed Packages": f"Your solution can use any relevant python packages. Feel free to use any other packages too (all packages are already installed!)."
        }
        return env_prompt

    @property
    def _prompt_impl_guideline(self):
        impl_guideline = [
            "The code should **implement the proposed solution** and **return the required results as in template solve function**. ",
            "The code should be a single-file python program that is self-contained and can be executed as-is. ",
            "No parts of the code should be skipped, don't terminate the before finishing the script. ",
            "Your response should only contain a single code block.",
            f"Be aware of the running time of the code, it should complete within {humanize.naturaldelta(self.cfg.exec.timeout)}. ",
            'You only need to implement the `def solve(**kwargs)` function. The data loading and evaluation are all done in the grading system. ',
        ]
        if self.acfg.expose_prediction:
            impl_guideline.append(
                "The implementation should include a solve(**kwargs) function, "
                "allowing users to seamlessly reuse the code to make predictions on new data. "
                "The solve function should always utilize **kwargs as input and get useful data from kwargs."
            )

        return {"Implementation guideline": impl_guideline}

    @property
    def _prompt_resp_fmt(self):
        return {
            "Response format": (
                "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
                "followed by a single markdown code block (wrapped in ```) which implements this solve function and return the required formatted output. "
                "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
            )
        }

    def plan_and_code_query(self, prompt, retries=3) -> tuple[str, str]:
        """Generate a natural language plan + code in the same LLM call and split them apart."""
        completion_text = None
        for _ in range(retries):
            completion_text = query(
                system_message=prompt,
                user_message=None,
                model=self.acfg.code.model,
                temperature=self.acfg.code.temp,
            )

            code = extract_code(completion_text)
            nl_text = extract_text_up_to_code(completion_text)

            if code and nl_text:
                # merge all code blocks into a single string
                return nl_text, code

            print("Plan + code extraction failed, retrying...")
        print("Final plan + code extraction attempt failed, giving up...")
        return "", completion_text  # type: ignore

    def _draft(self) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an expert in Operation Research problem and combinatorial optimization "
                "solving a combinatorial optimization problem. "
                "In order to solve this problem more effectively, you need to come up with an excellent and creative algorithm "
                "for a solution and then implement this solution in Python. We will now provide a description of the problem. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution sketch guideline": [
                "This first solution design should be relatively simple. ",
                "Take the Memory section into consideration when proposing the design,"
                " don't propose the same algorithm solution but keep the evaluation the same. ",
                "The solution sketch should be 3-5 sentences.",
                "Enclose all your code within a code block: ``` ... ``` and name the main function `def solve(**kwargs)`. "
                "Your function has a 10-second timeout; aim to return the best possible results within this limit.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline
        prompt["Instructions"] |= self._prompt_environment

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code)

    def _improve(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an expert in Operation Research problem and combinatorial optimization, "
                "solving a combinatorial optimization problem. "
                "You are provided with a previously developed "
                "solution below and should improve it in order to further increase the performance of algorithm. "
                "For this you should first outline a brief plan in natural language for how the algorithm can be improved and "
                "then implement this improvement in Python based on the provided previous solution. "
            ),
            "Task description": self.task_desc,
            "Memory": self.journal.generate_summary(),
            "Instructions": {},
        }
        prompt["Previous solution"] = {
            "Code": wrap_code(parent_node.code),
        }

        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Solution improvement sketch guideline": [
                "The solution sketch should be a brief natural language description of how the previous solution can be improved.",
                "You should be very specific and should only propose a single actionable improvement.",
                "This improvement should be atomic so that we can experimentally evaluate the effect of the proposed change.",
                "Take the Memory section into consideration when proposing the improvement.",
                "The solution sketch should be 3-5 sentences.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        plan, code = self.plan_and_code_query(prompt)
        return Node(
            plan=plan,
            code=code,
            parent=parent_node,
        )

    def _debug(self, parent_node: Node) -> Node:
        prompt: Any = {
            "Introduction": (
                "You are an expert in Operation Research problem and combinatorial optimization, "
                "solving a combinatorial optimization problem. "
                "Your previous solution had a bug, so based on the information below, you should revise it in order to fix this bug. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            ),
            "Task description": self.task_desc,
            "Previous (buggy) implementation": wrap_code(parent_node.code),
            "Execution output": wrap_code(parent_node.term_out, lang=""),
            "Instructions": {},
        }
        prompt["Instructions"] |= self._prompt_resp_fmt
        prompt["Instructions"] |= {
            "Bugfix improvement sketch guideline": [
                "You should write a brief natural language description (3-5 sentences) of how the issue in the previous implementation can be fixed.",
            ],
        }
        prompt["Instructions"] |= self._prompt_impl_guideline

        if self.acfg.data_preview:
            prompt["Data Overview"] = self.data_preview

        plan, code = self.plan_and_code_query(prompt)
        return Node(plan=plan, code=code, parent=parent_node)

    def draft(self):
        return self._draft()

    def improve(self, parent_node):
        return self._improve(parent_node)

    def debug(self, parent_node):
        return self._debug(parent_node)

    def update_data_preview(
            self,
    ):
        # self.data_preview = data_preview.generate(self.cfg.workspace_dir)
        self.data_preview = ""

    def step(self, exec_callback: ExecCallbackType):
        if not self.journal.nodes or self.data_preview is None:
            self.update_data_preview()

        parent_node = self.search_policy()
        logger.debug(f"Agent is generating code, parent node type: {type(parent_node)}")

        if parent_node is None:
            result_node = self._draft()
        elif parent_node.is_buggy:
            result_node = self._debug(parent_node)
        else:
            result_node = self._improve(parent_node)

        self.parse_exec_result(
            node=result_node,
            exec_result=exec_callback(result_node.code, True),
        )
        self.journal.append(result_node)

    def parse_exec_result(self, node: Node, exec_result: ExecutionResult):
        logger.info(f"Agent is parsing execution results for node {node.id}")

        node.absorb_exec_result(exec_result)

        prompt = {
            "Introduction": (
                "You are an expert in combinatorial optimization. "
                "You have written code to solve this task and now need to evaluate the output of the code execution. "
                "You should determine if there were any bugs as well as report the empirical findings."
            ),
            "Task description": self.task_desc,
            "Implementation": wrap_code(node.code),
            "Execution output": wrap_code(node.term_out, lang=""),
        }

        output_schema = {
            "name": "my_schema",
            "schema": {
                "type": "object",
                "properties": {
                    "is_bug": {
                        "type": "boolean",
                        "description": "true if the output log shows that the execution failed or has some bug, otherwise false. note that timeout is not a bug, but indicate the solution is not efficient enough to find results in limited time. so if you see timeout this should be false."
                    },
                    "summary": {
                        "type": "string",
                        "description": "if there is a bug, propose a fix. Otherwise, write a short summary (2-3 sentences) describing the empirical findings."
                    },
                    "metric": {
                        "type": "number",
                        "description": "If the code ran successfully, report the value of the validation metric. Otherwise, leave it null. Note that the score is normalized relative to the reference score, where a higher value is always better"
                    },
                    "lower_is_better": {
                        "type": "boolean",
                        "description": "this is always false. so if you see a very negative score or 0.0, it usually means the solution does not meet all the constraints, and a large penalty is applied."
                    }
                },
                "required": [
                    "is_bug",
                    "summary",
                    "metric",
                    "lower_is_better"
                ],
                "additionalProperties": False
            },
            "strict": True
        }

        response_format = {"type": "json_schema", "json_schema": output_schema}
        response = query(
                system_message=prompt,
                user_message=None,
                response_format=response_format,
                model=self.acfg.feedback.model,
                temperature=self.acfg.feedback.temp,
            )

        import json
        response = json.loads(response)
        print(response)

        # if the metric isn't a float then fill the metric with the worst metric
        if not isinstance(response["metric"], float):
            response["metric"] = None

        node.analysis = response["summary"]
        node.is_buggy = (
                response["is_bug"]
                or node.exc_type is not None
                or response["metric"] is None
        )

        if node.is_buggy:
            node.metric = WorstMetricValue()
        else:
            node.metric = MetricValue(
                response["metric"], maximize=not response["lower_is_better"]
            )
