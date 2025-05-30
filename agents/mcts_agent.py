import copy
import random
import math
import numpy as np
import json
from typing import Optional, Tuple, List, Dict, Any

# Import the original MCTS-AHD components
from agents.mcts.source.mcts import MCTS, MCTSNode
from agents.mcts.source.evolution import Evolution
from agents.mcts.source.getParas import Paras
from agents.mcts.utils.utils import init_client
import agents.mcts.source.prob_rank as prob_rank
import agents.mcts.source.pop_greedy as pop_greedy


class ExternalPrompts:
    """Simplified prompts interface for external problems"""

    def __init__(self, problem_description: str, func_name: str = "solve"):
        self.problem_description = problem_description
        self.func_name = func_name
        self.func_inputs = ["**kwargs"]
        self.func_outputs = ["a dict that same as defined in solve template"]

    def get_task(self):
        return self.problem_description

    def get_func_name(self):
        return self.func_name

    def get_func_inputs(self):
        return self.func_inputs

    def get_func_outputs(self):
        return self.func_outputs

    def get_inout_inf(self):
        return f"The {self.func_name} function takes problem-specific inputs and returns a solution."

    def get_other_inf(self):
        return ""


class MctsAhd:
    """
    MCTS-AHD wrapper that exactly follows the original implementation workflow
    but allows external evaluation of generated codes.

    Supports both maximization (higher score is better) and minimization (lower score is better) problems.

    Key fix: Bypasses InterfaceEC's synchronous evaluation and works directly with Evolution class
    """

    def __init__(self,
                 problem: str,
                 func_name: str = "solve",
                 score_type: str = "max",  # "max" if higher score is better, "min" if lower is better
                 model: str = "gpt-4o-mini",
                 temperature: float = 1.0,
                 api_key: str = "INPUT_YOUR_OPENAI_API",
                 pop_size: int = 10,
                 init_size: int = 4,
                 max_fe: int = 1000,
                 debug_mode: bool = False):
        """
        Initialize MCTS-AHD wrapper following the exact original structure

        Args:
            problem: Problem description
            func_name: Name of the function to generate
            score_type: "max" if higher score is better, "min" if lower score is better
            model: LLM model to use
            temperature: LLM temperature
            api_key: OpenAI API key
            pop_size: Population size
            init_size: Initial population size
            max_fe: Maximum function evaluations
            debug_mode: Debug mode flag
        """

        if score_type not in ["max", "min"]:
            raise ValueError("score_type must be 'max' or 'min'")

        self.score_type = score_type
        print(f"Initializing MCTS-AHD for {'maximization' if score_type == 'max' else 'minimization'} problem")

        # Set up parameters exactly like the original
        self.paras = Paras()
        self.paras.set_paras(
            method="mcts_ahd",
            init_size=init_size,
            pop_size=pop_size,
            llm_model=None,  # Will be set below
            ec_fe_max=max_fe,
            exp_output_path="./",
            exp_debug_mode=debug_mode,
            eva_timeout=60
        )

        # Create prompts interface
        self.prompts = ExternalPrompts(problem, func_name)

        # Set up LLM client (mimicking init_client behavior)
        class MockConfig(dict):
            """Dictionary that also supports attribute access."""

            # accept explicit arguments instead of relying on globals
            def __init__(self, model, temperature, api_key):
                super().__init__(model=model,
                                 temperature=temperature,
                                 api_key=api_key)

            # allow dot access → fallback to normal dict keys
            def __getattr__(self, item):
                try:
                    return self[item]
                except KeyError as exc:
                    raise AttributeError(item) from exc

        mock_cfg = MockConfig(model, temperature, api_key)
        client = init_client(mock_cfg)

        # Initialize Evolution class directly (bypass InterfaceEC)
        self.evolution = Evolution(
            api_endpoint="chat.openai.com",
            api_key=api_key,
            model_LLM=client,
            debug_mode=debug_mode,
            prompts=self.prompts,
            use_local_llm=False,
            url=None
        )

        # Initialize core components exactly like original MCTS_AHD.__init__
        self.select = prob_rank
        self.manage = pop_greedy

        # Core parameters from original
        self.init_size = self.paras.init_size
        self.pop_size = self.paras.pop_size
        self.fe_max = self.paras.ec_fe_max
        self.eval_times = 0

        # Exactly like original (from getParas.py)
        self.operators = ['e1', 'e2', 'm1', 'm2', 's1']  # Note: no 'i1' in main loop
        self.operator_weights = [0, 1, 2, 2, 1]  # e1=0, e2=1, m1=2, m2=2, s1=1
        self.m = 5  # From paras.ec_m = 5

        self.debug_mode = self.paras.exp_debug_mode
        self.timeout = self.paras.eva_timeout

        # State tracking
        self.state = "not_started"
        self.initialization_count = 0
        self.mcts = None
        self.nodes_set = []
        self.current_iteration = 0

        # Pending evaluation tracking
        self.pending_code = None
        self.pending_parents = None
        self.pending_operator = None
        self.pending_context = None  # 'init' or 'expand'

        print(
            f"MCTS-AHD wrapper initialized. Score type: {score_type} (higher is {'better' if score_type == 'max' else 'worse'})")

    def step(self) -> Optional[str]:
        """
        Advance one step in the MCTS-AHD algorithm
        Returns code to evaluate, or None if waiting for feedback or finished
        """
        if self.state == "waiting_for_feedback":
            raise ValueError("Waiting for feedback on previous code. Call feedback() first.")

        if self.state == "finished":
            return None

        if self.state == "not_started":
            self._initialize()

        if self.state == "initializing":
            return self._initialization_step()
        elif self.state == "evolving":
            return self._evolution_step()

        return None

    def _initialize(self):
        """Initialize the MCTS-AHD system exactly like the original"""
        print("- Initialization Start -")

        self.nodes_set = []
        self.mcts = MCTS('Root')
        self.state = "initializing"

    def _initialization_step(self) -> Optional[str]:
        """Handle initialization phase exactly like original"""
        if self.initialization_count == 0:
            # First algorithm using i1
            return self._generate_with_evolution_operator("i1", [], "init")
        else:
            # Subsequent algorithms using e1
            parents = self._select_parents_for_operator("e1", self.nodes_set, None)
            return self._generate_with_evolution_operator("e1", parents, "init")

    def _evolution_step(self) -> Optional[str]:
        """Handle evolution phase exactly like original main loop"""
        if self.eval_times >= self.fe_max:
            self.state = "finished"
            return None

        print(f"Current performances of MCTS nodes: {getattr(self.mcts, 'rank_list', [])}")

        # Tree traversal with UCT (exactly like original)
        cur_node = self.mcts.root
        while len(cur_node.children) > 0 and cur_node.depth < self.mcts.max_depth:
            uct_scores = [self.mcts.uct(node, max(1 - self.eval_times / self.fe_max, 0))
                          for node in cur_node.children]
            selected_pair_idx = uct_scores.index(max(uct_scores))

            # Check if we should expand at current node (exactly like original)
            if int((cur_node.visits) ** self.mcts.alpha) > len(cur_node.children):
                if cur_node == self.mcts.root:
                    op = 'e1'
                    return self._expand_node(cur_node, op)
                else:
                    i = 1  # Fixed to 1 like original (skips e1 since weight=0)
                    op = self.operators[i]  # This will be 'e2'
                    return self._expand_node(cur_node, op)

            cur_node = cur_node.children[selected_pair_idx]

        # Apply all operators at selected node (exactly like original)
        n_op = len(self.operators)
        for i in range(n_op):
            op = self.operators[i]
            op_w = self.operator_weights[i]
            if op_w > 0:  # Only if weight > 0 (skips e1 which has weight 0)
                print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}", end="|")
                for j in range(op_w):
                    return self._expand_node(cur_node, op)

        # If we get here, move to next iteration
        return self._evolution_step()

    def _generate_with_evolution_operator(self, operator: str, parents: List, context: str) -> Optional[str]:
        """Generate code using Evolution class directly"""
        try:
            print(f"Generating with operator: {operator}, parents: {len(parents)}")

            # Call Evolution methods directly (bypass InterfaceEC)
            if operator == "i1":
                code, algorithm = self.evolution.i1()
            elif operator == "e1":
                code, algorithm = self.evolution.e1(parents)
            elif operator == "e2":
                code, algorithm = self.evolution.e2(parents)
            elif operator == "m1":
                code, algorithm = self.evolution.m1(parents[0] if parents else None)
            elif operator == "m2":
                code, algorithm = self.evolution.m2(parents[0] if parents else None)
            elif operator == "s1":
                code, algorithm = self.evolution.s1(parents)
            else:
                raise ValueError(f"Unknown operator: {operator}")

            # Post-process algorithm description
            algorithm_desc = self.evolution.post_thought(code, algorithm)

            # Store pending state
            self.pending_code = code
            self.pending_parents = parents
            self.pending_operator = operator
            self.pending_context = context
            self.pending_algorithm = algorithm_desc
            self.state = "waiting_for_feedback"

            return code

        except Exception as e:
            print(f"Error generating code with operator {operator}: {e}")
            # Return a simple fallback
            fallback_code = f"""
def {self.prompts.func_name}(problem_data):
    # Fallback heuristic for {self.prompts.problem_description}
    return 1.0
"""
            self.pending_code = fallback_code
            self.pending_parents = parents
            self.pending_operator = operator
            self.pending_context = context
            self.pending_algorithm = "Fallback heuristic"
            self.state = "waiting_for_feedback"
            return fallback_code

    def _expand_node(self, cur_node: MCTSNode, operator: str) -> Optional[str]:
        """Expand MCTS node exactly like original expand() method"""
        # Select parents based on operator type (exactly like original)
        parents = self._select_parents_for_operator(operator, self.nodes_set, cur_node)

        # Store expansion context
        self.pending_node = cur_node
        self.pending_expansion_operator = operator

        return self._generate_with_evolution_operator(operator, parents, "expand")

    def _select_parents_for_operator(self, operator: str, nodes_set: List, cur_node: Optional[MCTSNode]) -> List:
        """Select parents for operator exactly like original expand() method"""
        if operator == 's1':
            # Path-based selection for s1
            if cur_node is None or cur_node.code == "Root":
                return nodes_set
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = self.manage.population_management_s1(path_set, len(path_set))
            return path_set

        elif operator == 'e1':
            # Selection from subtrees (exactly like original)
            if cur_node == self.mcts.root:
                e1_set = []
                for children in self.mcts.root.children:
                    if hasattr(children, 'subtree') and len(children.subtree) > 0:
                        selected = random.choices(range(len(children.subtree)), k=1)[0]
                        e1_set.append(copy.deepcopy(children.subtree[selected].raw_info))
                return e1_set if e1_set else nodes_set
            else:
                # Use rank-based selection for non-root e1
                m = min(random.randint(2, self.m), len(nodes_set)) if nodes_set else 0
                return self.select.parent_selection_e1(nodes_set, m) if m > 0 else []

        elif operator == 'e2':
            # Two-parent selection (best + other)
            if len(nodes_set) < 2:
                return nodes_set
            sorted_pop = sorted(nodes_set, key=lambda x: x['objective'])
            good_parent = sorted_pop[0]
            other_parent = random.choice([p for p in nodes_set if p != good_parent])
            return [good_parent, other_parent]

        elif operator in ['m1', 'm2']:
            # Single parent selection
            if not nodes_set:
                return []
            return self.select.parent_selection(nodes_set, 1)

        else:
            # Default: use current population
            return nodes_set

    def feedback(self, score: float, description: str = ""):
        """
        Provide feedback on the last generated code

        Args:
            score: Performance score from external evaluation
                  For score_type="max": higher score is better
                  For score_type="min": lower score is better
            description: Optional description
        """
        if self.state != "waiting_for_feedback":
            raise ValueError("No pending code to provide feedback for")

        if self.pending_code is None:
            raise ValueError("No pending code found")

        # Convert score to internal objective (exactly like original problem_adapter.py line 165-166)
        if self.score_type == "max":
            objective = -score  # Higher score → lower objective (better for minimization)
        else:
            objective = score  # Lower score → lower objective (consistent)

        print(
            f"Received feedback: score={score} ({'higher=better' if self.score_type == 'max' else 'lower=better'}), internal_objective={objective}")

        # Create individual exactly like original
        individual = {
            'algorithm': getattr(self, 'pending_algorithm', 'Generated algorithm'),
            'code': self.pending_code,
            'objective': objective,
            'other_inf': description
        }

        # Handle feedback based on context
        if self.pending_context == "init":
            self._complete_initialization(individual)
        elif self.pending_context == "expand":
            self._complete_expansion(individual)

        # Clear pending state
        self._clear_pending_state()

        # Increment evaluation counter
        self.eval_times += 1

    def _complete_initialization(self, individual: Dict):
        """Complete initialization step with feedback"""
        # Add to population
        self.nodes_set.append(individual)

        # Create MCTS node exactly like original (mcts_ahd.py line 89-90)
        nownode = MCTSNode(
            individual['algorithm'], individual['code'], individual['objective'],
            parent=self.mcts.root, depth=1, visit=1,
            Q=-1 * individual['objective'],  # Q = -objective (higher Q is better)
            raw_info=individual
        )

        self.mcts.root.add_child(nownode)
        if not hasattr(self.mcts.root, 'children_info'):
            self.mcts.root.children_info = []
        self.mcts.root.children_info.append(individual)

        self.mcts.backpropagate(nownode)
        nownode.subtree = [nownode]  # Initialize subtree like original

        self.initialization_count += 1

        # Check if initialization complete
        if self.initialization_count >= self.init_size:
            # Manage population and switch to evolution
            size_act = min(len(self.nodes_set), self.pop_size)
            self.nodes_set = self.manage.population_management(self.nodes_set, size_act)
            print("- Initialization Finished - Evolution Start -")
            self.state = "evolving"
        else:
            self.state = "initializing"

    def _complete_expansion(self, individual: Dict):
        """Complete expansion operation with feedback"""
        if not hasattr(self, 'pending_node'):
            self.state = "evolving"
            return

        cur_node = self.pending_node
        option = self.pending_expansion_operator

        # Handle duplicate detection for e1 (exactly like original)
        if option == 'e1' and cur_node == self.mcts.root:
            duplicate = False
            for child_info in getattr(self.mcts.root, 'children_info', []):
                if abs(child_info['objective'] - individual['objective']) < 1e-10:
                    duplicate = True
                    break

            if duplicate:
                print(f"Duplicated e1, no action, Father is Root, Abandon Obj: {individual['objective']}")
                self.state = "evolving"
                return
            else:
                print(f"Action: {option}, Father is Root, Now Obj: {individual['objective']}")
        else:
            father_obj = cur_node.raw_info['objective'] if hasattr(cur_node, 'raw_info') else 'N/A'
            print(
                f"Action: {option}, Father Obj: {father_obj}, Now Obj: {individual['objective']}, Depth: {cur_node.depth + 1}")

        # Add to population and create MCTS node (exactly like original expand())
        if individual['objective'] != float('inf'):
            # Add to population
            self.nodes_set.append(individual)
            size_act = min(len(self.nodes_set), self.pop_size)
            self.nodes_set = self.manage.population_management(self.nodes_set, size_act)

            # Create new MCTS node exactly like original
            nownode = MCTSNode(
                individual['algorithm'], individual['code'], individual['objective'],
                parent=cur_node, depth=cur_node.depth + 1, visit=1,
                Q=-1 * individual['objective'], raw_info=individual
            )

            if option == 'e1':
                nownode.subtree = [nownode]

            cur_node.add_child(nownode)
            if not hasattr(cur_node, 'children_info'):
                cur_node.children_info = []
            cur_node.children_info.append(individual)

            self.mcts.backpropagate(nownode)

        # Save population like original
        try:
            filename = f"population_generation_{self.eval_times}.json"
            with open(filename, 'w') as f:
                json.dump(self.nodes_set, f, indent=5)
        except:
            pass  # Ignore save errors

        self.current_iteration += 1
        self.state = "evolving"

    def _clear_pending_state(self):
        """Clear all pending state variables"""
        self.pending_code = None
        self.pending_parents = None
        self.pending_operator = None
        self.pending_context = None

        # Clear expansion-specific pending state
        for attr in ['pending_node', 'pending_expansion_operator', 'pending_algorithm']:
            if hasattr(self, attr):
                delattr(self, attr)

    def get_best_algorithm(self) -> Optional[Dict]:
        """Get the best algorithm found so far"""
        if not self.nodes_set:
            return None
        # Best is minimum objective (works for both max and min problems after conversion)
        return min(self.nodes_set, key=lambda x: x['objective'])

    def get_best_score(self) -> Optional[float]:
        """Get the best score in original user units"""
        best = self.get_best_algorithm()
        if best is None:
            return None

        # Convert internal objective back to user score
        objective = best['objective']
        if self.score_type == "max":
            return -objective  # Convert back: score = -objective
        else:
            return objective  # score = objective

    def get_stats(self) -> Dict:
        """Get current statistics"""
        stats = {
            "state": self.state,
            "score_type": self.score_type,
            "evaluations": self.eval_times,
            "max_evaluations": self.fe_max,
            "num_algorithms": len(self.nodes_set),
            "initialization_count": self.initialization_count,
            "required_init": self.init_size
        }

        if self.nodes_set:
            best_score = self.get_best_score()
            best_obj = self.get_best_algorithm()['objective']
            stats.update({
                "best_score": best_score,
                "best_objective": best_obj,
                "score_interpretation": f"{'higher' if self.score_type == 'max' else 'lower'} is better"
            })

        if self.mcts and self.mcts.root.children:
            stats["mcts_nodes"] = len(self.mcts.root.children)
            stats["mcts_rank_list"] = getattr(self.mcts, 'rank_list', [])

        return stats

    def is_finished(self) -> bool:
        """Check if algorithm has finished"""
        return self.state == "finished" or self.eval_times >= self.fe_max

    def get_generated_code_info(self) -> Dict:
        """Get information about the currently pending code for evaluation"""
        if self.pending_code is None:
            return {}

        return {
            "function_name": self.prompts.func_name,
            "function_inputs": self.prompts.func_inputs,
            "function_outputs": self.prompts.func_outputs,
            "function_description": self.prompts.get_inout_inf(),
            "problem_description": self.prompts.problem_description,
            "operator_used": self.pending_operator,
            "context": self.pending_context
        }
