import numpy as np
import json
import random
import time
import heapq
from typing import Optional, Dict, Any, List

# Import EoH components
from agents.eoh.methods.eoh.eoh_evolution import Evolution
from agents.eoh.llm.interface_LLM import InterfaceLLM



class SimpleProblem:
    """A simple problem wrapper for custom problems defined by text description"""

    def __init__(self, problem_description: str):
        self.problem_description = problem_description
        self.prompts = SimplePrompts(problem_description)

    def evaluate(self, code: str) -> float:
        """Placeholder evaluation - will be handled by external feedback"""
        return 0.0


class SimplePrompts:
    """Simple prompts wrapper for custom problems"""

    def __init__(self, problem_description: str):
        self.problem_description = problem_description

    def get_task(self) -> str:
        return f"Solve the following combinatorial optimization problem: {self.problem_description}"

    def get_func_name(self) -> str:
        return "solve"

    def get_func_inputs(self) -> list:
        return ["**kwargs"]

    def get_func_outputs(self) -> list:
        return ["a dict that same as defined in solve template"]

    def get_inout_inf(self) -> str:
        return "The function should take a problem instance and return an optimized solution."

    def get_other_inf(self) -> str:
        return "Focus on creating an efficient and effective algorithm for this optimization problem."


class EoHAgent:
    """
    A real-time wrapper around EoH for step-by-step interaction.

    Generates algorithms one at a time following original EoH workflow:
    1. Initial phase: Generate 2*pop_size algorithms using i1 (step by step)
    2. Evolution phase: Use evolutionary operators e1, e2, m1, m2 in sequence

    Usage:
        agent = EoHAgent(problem="Traveling Salesman Problem optimization")
        code = agent.step()  # Get a new algorithm (real-time generation)
        agent.feedback(0.6, "Good performance but can be improved")  # Provide feedback
    """

    def __init__(
            self,
            problem: str,
            llm_api_endpoint: str = "api.openai.com",
            llm_api_key: str = "your-key-here",
            llm_model: str = "gpt-3.5-turbo",
            pop_size: int = 5,
            debug_mode: bool = False,
            maximize: bool = False,  # Original EoH is minimization by default!
            operators: List[str] = None,
            operator_weights: List[float] = None,
            m: int = 2  # number of parents for crossover operations
    ):
        """
        Initialize the EoH Agent.

        Args:
            problem: Text description of the combinatorial optimization problem
            llm_api_endpoint: LLM API endpoint
            llm_api_key: LLM API key
            llm_model: LLM model name
            pop_size: Population size for evolution
            debug_mode: Whether to enable debug mode
            maximize: If True, higher scores are better; if False, lower scores are better (DEFAULT: False for original EoH minimization)
            operators: List of evolutionary operators (default: ['e1','e2','m1','m2'])
            operator_weights: Probability weights for each operator (default: [1.0, 1.0, 1.0, 1.0])
            m: Number of parents for crossover operations (default: 2)
        """
        self.problem_description = problem
        self.pop_size = pop_size
        self.debug_mode = debug_mode
        self.maximize = maximize
        self.m = m

        # Set default operators and weights (following original EoH)
        self.operators = operators if operators is not None else ['e1', 'e2', 'm1', 'm2']
        self.operator_weights = operator_weights if operator_weights is not None else [1.0] * len(self.operators)

        if len(self.operator_weights) != len(self.operators):
            print("Warning! Lengths of operator_weights and operators should be the same.")
            self.operator_weights = [1.0] * len(self.operators)

        # Initialize problem wrapper
        self.problem = SimpleProblem(problem)

        # Initialize evolution engine
        self.evolution = Evolution(
            llm_api_endpoint,
            llm_api_key,
            llm_model,
            llm_use_local=False,
            llm_local_url=None,
            debug_mode=debug_mode,
            prompts=self.problem.prompts
        )

        # Initialize population and state
        self.population = []
        self.generation = 0
        self.current_individual = None
        self.current_operator_idx = 0
        self.initial_phase = True  # Track if we're still in initial population phase
        self.initial_count = 0  # Count initial algorithms generated

        # Set random seed for reproducibility
        random.seed(2024)

    def _population_management(self, pop: List[Dict], size: int) -> List[Dict]:
        """
        Population management following original EoH pop_greedy.py implementation exactly
        """
        # Filter out individuals without objective scores (exact original logic)
        pop = [individual for individual in pop if individual['objective'] is not None]

        if size > len(pop):
            size = len(pop)

        # Remove duplicates based on objective values (exact original logic)
        unique_pop = []
        unique_objectives = []
        for individual in pop:
            if individual['objective'] not in unique_objectives:
                unique_pop.append(individual)
                unique_objectives.append(individual['objective'])

        # Select best individuals based on optimization direction
        if self.maximize:
            # For maximization: keep individuals with highest scores (adapted for user needs)
            pop_new = heapq.nlargest(size, unique_pop, key=lambda x: x['objective'])
        else:
            # For minimization: keep individuals with lowest scores (original EoH behavior)
            pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])

        return pop_new

    def _parent_selection(self, pop: List[Dict], m: int) -> List[Dict]:
        """
        Parent selection following original EoH prob_rank.py implementation exactly

        IMPORTANT: Original assumes population is already sorted with best individuals first
        (which happens after population_management via heapq.nsmallest/nlargest)
        """
        if not pop or m <= 0:
            return []

        # Filter valid individuals
        valid_pop = [ind for ind in pop if ind['objective'] is not None]
        if not valid_pop:
            return []

        # Original EoH assumes population is already sorted by population_management
        # No additional sorting needed here - trust the population order

        # Original EoH rank-based selection (exact implementation)
        ranks = [i for i in range(len(valid_pop))]
        probs = [1 / (rank + 1 + len(valid_pop)) for rank in ranks]

        # Limit selection to available individuals
        m = min(m, len(valid_pop))

        try:
            parents = random.choices(valid_pop, weights=probs, k=m)
            return parents
        except:
            return random.choices(valid_pop, k=m)

    def _generate_algorithm(self, operator: str) -> Optional[Dict]:
        """
        Generate a single algorithm using the specified operator
        """
        try:
            offspring = {
                'algorithm': None,
                'code': None,
                'objective': None,
                'other_inf': None
            }

            if operator == "i1":
                code, algorithm = self.evolution.i1()
                offspring['code'] = code
                offspring['algorithm'] = algorithm

            elif operator == "e1":
                if len(self.population) >= self.m:
                    parents = self._parent_selection(self.population, self.m)
                    if len(parents) >= self.m:
                        code, algorithm = self.evolution.e1(parents)
                        offspring['code'] = code
                        offspring['algorithm'] = algorithm
                    else:
                        return None  # Not enough parents
                else:
                    return None  # Not enough population

            elif operator == "e2":
                if len(self.population) >= self.m:
                    parents = self._parent_selection(self.population, self.m)
                    if len(parents) >= self.m:
                        code, algorithm = self.evolution.e2(parents)
                        offspring['code'] = code
                        offspring['algorithm'] = algorithm
                    else:
                        return None
                else:
                    return None

            elif operator == "m1":
                if len(self.population) >= 1:
                    parents = self._parent_selection(self.population, 1)
                    if len(parents) >= 1:
                        code, algorithm = self.evolution.m1(parents[0])
                        offspring['code'] = code
                        offspring['algorithm'] = algorithm
                    else:
                        return None
                else:
                    return None

            elif operator == "m2":
                if len(self.population) >= 1:
                    parents = self._parent_selection(self.population, 1)
                    if len(parents) >= 1:
                        code, algorithm = self.evolution.m2(parents[0])
                        offspring['code'] = code
                        offspring['algorithm'] = algorithm
                    else:
                        return None
                else:
                    return None

            elif operator == "m3":
                if len(self.population) >= 1:
                    parents = self._parent_selection(self.population, 1)
                    if len(parents) >= 1:
                        code, algorithm = self.evolution.m3(parents[0])
                        offspring['code'] = code
                        offspring['algorithm'] = algorithm
                    else:
                        return None
                else:
                    return None
            else:
                print(f"Evolution operator [{operator}] has not been implemented!")
                return None

            return offspring

        except Exception as e:
            if self.debug_mode:
                print(f"Error generating algorithm with operator {operator}: {e}")
            return None

    def _should_continue_initial_phase(self) -> bool:
        """
        Determine if we should continue initial population generation
        Original EoH creates 2 * pop_size initial algorithms before main evolution
        """
        return self.initial_phase and self.initial_count < (2 * self.pop_size)

    def step(self) -> str:
        """
        Generate and return a new algorithm code in real-time.
        Follows original EoH workflow: initial population first, then main evolution.

        Returns:
            str: Python code for a new algorithm
        """
        try:
            # Phase 1: Initial population generation (like original population_generation)
            if self._should_continue_initial_phase():
                individual = self._generate_algorithm("i1")

                if individual is not None:
                    self.current_individual = individual
                    self.initial_count += 1

                    if self.debug_mode:
                        print(
                            f"Generated initial algorithm {self.initial_count}/{2 * self.pop_size}: {individual['algorithm']}")

                    # Check if initial phase is complete
                    if not self._should_continue_initial_phase():
                        self.initial_phase = False
                        if self.debug_mode:
                            print("Initial population phase complete, starting main evolution")

                    return individual['code']

            # Phase 2: Main evolution loop (like original EoH main loop)
            else:
                # Generate algorithm using evolutionary operators (following original EoH workflow)
                attempts = 0
                max_attempts = len(self.operators) * 3  # Allow multiple cycles through operators

                while attempts < max_attempts:
                    # Get current operator (cycle through in sequence like original)
                    operator_idx = self.current_operator_idx % len(self.operators)
                    operator = self.operators[operator_idx]
                    operator_weight = self.operator_weights[operator_idx]

                    if self.debug_mode:
                        print(f"Trying operator {operator} (weight: {operator_weight})")

                    # Check probability (following original EoH logic: if (np.random.rand() < op_w))
                    if random.random() < operator_weight:
                        individual = self._generate_algorithm(operator)

                        if individual is not None:
                            self.current_individual = individual

                            if self.debug_mode:
                                print(f"Generated algorithm using {operator}: {individual['algorithm']}")

                            # Move to next operator for next call (like original sequence)
                            self.current_operator_idx += 1

                            return individual['code']
                        else:
                            if self.debug_mode:
                                print(f"Failed to generate algorithm with {operator} (insufficient population)")
                    else:
                        if self.debug_mode:
                            print(f"Skipped operator {operator} (probability check failed)")

                    # Move to next operator and try again
                    self.current_operator_idx += 1
                    attempts += 1

                # Fallback: generate using i1
                if self.debug_mode:
                    print("Fallback to i1 generation")

                individual = self._generate_algorithm("i1")
                if individual is not None:
                    self.current_individual = individual
                    return individual['code']

            # Final fallback
            return """
import numpy as np

def solve(**kwargs):
    # Simple fallback solution
    return kwargs
"""

        except Exception as e:
            if self.debug_mode:
                print(f"Error in step(): {e}")
            # Return a simple fallback algorithm
            return """
import numpy as np

def solve(**kwargs):
    # Simple fallback solution
    return kwargs
"""

    def feedback(self, score: float, description: str = "") -> None:
        """
        Provide feedback on the last generated algorithm.

        Args:
            score: Numerical score for the algorithm
            description: Optional text description of the performance
        """
        if self.current_individual is None:
            if self.debug_mode:
                print("Warning: No current individual to provide feedback for")
            return

        # Update the current individual with the score
        self.current_individual['objective'] = score
        self.current_individual['other_inf'] = description

        # Add to population (following original add2pop logic exactly)
        # Original EoH checks for duplicates but adds anyway (just warns)
        for ind in self.population:
            if ind['objective'] == self.current_individual['objective']:
                if self.debug_mode:
                    print("duplicated result, retrying ... ")
                break

        # Always add (original behavior, even if duplicate detected)
        self.population.append(self.current_individual.copy())

        # Apply population management (following original EoH logic)
        if len(self.population) > 0:
            size_act = min(len(self.population), self.pop_size)
            self.population = self._population_management(self.population, size_act)

        if self.debug_mode:
            print(f"Feedback received - Score: {score}, Description: {description}")
            print(f"Population size: {len(self.population)}")
            if len(self.population) > 0:
                valid_scores = [ind['objective'] for ind in self.population if ind['objective'] is not None]
                if valid_scores:
                    if self.maximize:
                        best_score = max(valid_scores)
                    else:
                        best_score = min(valid_scores)
                    print(f"Best score so far: {best_score}")

        # Clear current individual
        self.current_individual = None
        self.generation += 1

    def get_best_algorithm(self) -> Optional[Dict[str, Any]]:
        """
        Get the best algorithm found so far.

        Returns:
            Dict containing the best algorithm's code, description, and score, or None if no algorithms evaluated yet
        """
        if not self.population:
            return None

        # Find the individual with the best objective score
        valid_individuals = [ind for ind in self.population if ind['objective'] is not None]
        if not valid_individuals:
            return None

        if self.maximize:
            best_individual = max(valid_individuals, key=lambda x: x['objective'])
        else:
            best_individual = min(valid_individuals, key=lambda x: x['objective'])

        return {
            'code': best_individual['code'],
            'algorithm': best_individual['algorithm'],
            'score': best_individual['objective'],
            'description': best_individual.get('other_inf', '')
        }

    def get_population_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the current population.

        Returns:
            Dict containing population statistics
        """
        if not self.population:
            return {
                'size': 0,
                'evaluated': 0,
                'generation': self.generation,
                'initial_phase': self.initial_phase,
                'initial_progress': f"{self.initial_count}/{2 * self.pop_size}" if self.initial_phase else "Complete"
            }

        evaluated = [ind for ind in self.population if ind['objective'] is not None]

        summary = {
            'size': len(self.population),
            'evaluated': len(evaluated),
            'generation': self.generation,
            'initial_phase': self.initial_phase,
            'initial_progress': f"{self.initial_count}/{2 * self.pop_size}" if self.initial_phase else "Complete"
        }

        if evaluated:
            scores = [ind['objective'] for ind in evaluated]
            if self.maximize:
                summary.update({
                    'best_score': max(scores),
                    'worst_score': min(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                })
            else:
                summary.update({
                    'best_score': min(scores),
                    'worst_score': max(scores),
                    'mean_score': np.mean(scores),
                    'std_score': np.std(scores)
                })

        return summary

    def save_state(self, filepath: str) -> None:
        """Save the current state of the agent to a file."""
        state = {
            'problem_description': self.problem_description,
            'population': self.population,
            'generation': self.generation,
            'current_operator_idx': self.current_operator_idx,
            'initial_phase': self.initial_phase,
            'initial_count': self.initial_count,
            'maximize': self.maximize,
            'operators': self.operators,
            'operator_weights': self.operator_weights,
            'm': self.m
        }

        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)

    def load_state(self, filepath: str) -> None:
        """Load the agent state from a file."""
        with open(filepath, 'r') as f:
            state = json.load(f)

        self.population = state['population']
        self.generation = state['generation']
        self.current_operator_idx = state['current_operator_idx']
        self.initial_phase = state.get('initial_phase', False)
        self.initial_count = state.get('initial_count', 0)


# Example usage
if __name__ == "__main__":
    # Initialize agent with OpenAI API
    # NOTE: Original EoH uses minimization by default (lower scores = better)
    # Set maximize=True if your problem has higher scores = better
    agent = EoHAgent(
        problem="Traveling Salesman Problem: find the shortest route visiting all cities exactly once",
        llm_api_endpoint="api.openai.com",
        llm_api_key="sk-your-openai-api-key-here",
        llm_model="gpt-3.5-turbo",
        maximize=False,  # Original EoH default (minimization)
        pop_size=5,  # Original default
        operators=['e1', 'e2', 'm1', 'm2'],  # Original default
        operator_weights=[1.0, 1.0, 1.0, 1.0],  # Original default
        m=2,  # Original default
        debug_mode=True
    )

    # Generate and evaluate algorithms in real-time
    for iteration in range(15):  # Increased to show both phases
        print(f"\n--- Iteration {iteration + 1} ---")

        # Get a new algorithm (real-time generation)
        code = agent.step()
        print(f"Generated code:\n{code[:200]}...")  # Show first 200 chars

        # Simulate evaluation
        # For TSP (minimization): lower route length = better performance
        simulated_score = random.uniform(100, 1000)  # Route length
        description = f"Route length: {simulated_score:.1f}"

        # Provide feedback
        agent.feedback(simulated_score, description)

        # Check progress
        summary = agent.get_population_summary()
        print(f"Population summary: {summary}")

        best = agent.get_best_algorithm()
        if best:
            print(f"Best algorithm so far - Score: {best['score']:.3f} (lower is better)")

    # Get the final best algorithm
    best_algorithm = agent.get_best_algorithm()
    if best_algorithm:
        print(f"\nFinal best algorithm:")
        print(f"Score: {best_algorithm['score']} (lower = better route)")
        print(f"Description: {best_algorithm['algorithm']}")

    # EXAMPLE FOR MAXIMIZATION PROBLEMS:
    # If your task has "higher scores = better", initialize with maximize=True:
    #
    # agent_max = EoHAgent(
    #     problem="Portfolio optimization: maximize returns",
    #     maximize=True,  # Higher scores = better
    #     llm_api_endpoint="api.openai.com",
    #     llm_api_key="your-key",
    #     llm_model="gpt-3.5-turbo"
    # )
    #
    # Then use normally - higher scores will be treated as better