import os
import logging
import numpy as np
import tempfile
import subprocess
import re
from typing import Dict, Optional, Tuple, List, Any
from omegaconf import OmegaConf
from dataclasses import dataclass
from typing import Optional
from functools import partial
import os


def call_llm(messages, model='openai/gpt-4o', reasoning_effort=None, n=None):
    import litellm
    from litellm import completion
    litellm.drop_params = True
    response = completion(model=model, messages=messages, reasoning_effort=reasoning_effort, n=n)
    if n == 1:
        return response.choices[0].message.content
    else:
        return [choice.message.content for choice in response.choices]


# Import necessary components for ReEvo-based approach
def extract_code_from_generator(content):
    """Extract code from the response of the code generator."""
    pattern_code = r'```python(.*?)```'
    code_string = re.search(pattern_code, content, re.DOTALL)
    code_string = code_string.group(1).strip() if code_string is not None else None
    if code_string is None:
        # Find the line that starts with "def" and the line that starts with "return", and extract the code in between
        lines = content.split('\n')
        start = None
        end = None
        for i, line in enumerate(lines):
            if line.startswith('def'):
                start = i
            if 'return' in line:
                end = i
                break
        if start is not None and end is not None:
            code_string = '\n'.join(lines[start:end + 1])

    if code_string is None:
        return None
    # Add import statements if not present
    if "np" in code_string:
        code_string = "import numpy as np\n" + code_string
    if "torch" in code_string:
        code_string = "import torch\n" + code_string
    return code_string


def filter_code(code_string):
    """Remove lines containing signature and import statements."""
    lines = code_string.split('\n')
    filtered_lines = []
    for line in lines:
        if line.startswith('def'):
            continue
        elif line.startswith('import'):
            continue
        elif line.startswith('from'):
            continue
        elif line.startswith('return'):
            filtered_lines.append(line)
            break
        else:
            filtered_lines.append(line)
    code_string = '\n'.join(filtered_lines)
    return code_string


def filter_traceback(s):
    """Extract traceback from stdout."""
    lines = s.split('\n')
    filtered_lines = []
    for i, line in enumerate(lines):
        if line.startswith('Traceback'):
            for j in range(i, len(lines)):
                filtered_lines.append(lines[j])
            return '\n'.join(filtered_lines)
    return ''  # Return an empty string if no Traceback is found


class ReEvoAgent:
    def __init__(
            self,
            problem_description: str,
            func_name: str = "solve",
            obj_type: str = "max",  # "max" or "min"
            llm_model: str = "gpt-3.5-turbo",
            temperature: float = 1.0,
            population_size: int = 10,  # Original default from paper
            init_population_size: int = 30,  # Original default from paper
            mutation_rate: float = 0.5,
            api_key: Optional[str] = None,
            base_url: Optional[str] = None,
            call_llm=None
    ):
        """
        Initialize the ReEvo agent.

        Args:
            problem_description: Description of the problem to solve
            func_name: Name of the function to generate (default: "solve")
            obj_type: Type of optimization objective ("max" or "min")
            llm_model: LLM model to use
            temperature: Temperature for LLM sampling
            population_size: Population size for evolutionary algorithm
            init_population_size: Initial population size
            mutation_rate: Mutation rate
            api_key: API key for LLM (if None, will use environment variable)
            base_url: Base URL for LLM API (if None, will use default)
            call_llm: Function to call LLM (must accept messages, return contents)
        """
        # Set up environment
        if api_key is not None:
            os.environ["OPENAI_API_KEY"] = api_key

        # Create temp directory for outputs
        self.workspace = tempfile.mkdtemp(prefix="reevo_agent_")
        # os.chdir(self.workspace)

        # Store parameters
        self.problem_description = problem_description
        self.func_name = func_name
        self.obj_type = obj_type
        self.llm_model = llm_model
        self.temperature = temperature

        # Create config
        self.cfg = self._create_config(
            population_size=population_size,
            init_population_size=init_population_size,
            mutation_rate=mutation_rate,
            base_url=base_url
        )

        # LLM interface
        self.call_llm = call_llm

        # State variables
        self.iteration = 0
        self.function_evals = 0
        self.population = []
        self.elitist = None
        self.long_term_reflection_str = ""
        self.solution_counter = 0
        self.pending_solutions = {}  # Map solution_id -> individual

        # Load prompts
        self._init_prompts()

        # Initialize population
        self._init_population()

    def _create_config(self, population_size, init_population_size, mutation_rate, base_url):
        """Create a config similar to the one used in ReEvo."""
        cfg = {
            "algorithm": "reevo",
            "pop_size": population_size,
            "init_pop_size": init_population_size,
            "mutation_rate": mutation_rate,
            "max_fe": 1000,  # Very large number, we'll control manually
            "timeout": 20,
            "diversify_init_pop": True,
            "problem": {
                "problem_name": "custom",
                "problem_type": "general",
                "obj_type": self.obj_type,
                "problem_size": 0,
                "func_name": self.func_name,
                "description": self.problem_description
            },
            "llm_client": {
                "model": self.llm_model,
                "temperature": self.temperature,
                "base_url": base_url
            }
        }
        return OmegaConf.create(cfg)

    def _init_prompts(self):
        """Initialize prompts for ReEvo."""
        # System prompts
        self.system_generator_prompt = """You are an expert in the domain of optimization and algorithm design. Your task is to design high-quality algorithms that can effectively solve specific problems.
Your response outputs Python code and nothing else. Format your code as a Python code string: "```python ... ```"."""

        self.system_reflector_prompt = """You are an expert in the domain of optimization and algorithm design. Your task is to give hints to design better solutions."""

        # User prompts
        self.user_generator_prompt = f"""Write a {self.func_name} function for the following problem:
{self.problem_description}

The function should follow this signature:
def {self.func_name}(**kwargs):
    # Your code here
    return # required output format
"""

        # Reflection prompts
        self.user_reflector_st_prompt = f"""Below are two {self.func_name} functions for the following problem:
{self.problem_description}

You are provided with two code versions below, where the second version performs better than the first one.

[Worse code]
{{worse_code}}

[Better code]
{{better_code}}

[Worse code performance]
{{worse_performance}}

[Better code performance]
{{better_performance}}

You respond with some hints for designing better solutions, based on the two code versions and their performance feedback. Use less than 50 words."""

        self.user_reflector_lt_prompt = f"""Below is your prior long-term reflection on designing solutions for the following problem:
{self.problem_description}

{{prior_reflection}}

Below are some newly gained insights.
{{new_reflection}}

Write constructive hints for designing better solutions, based on prior reflections and new insights and using less than 50 words."""

        # Crossover and mutation prompts
        self.crossover_prompt = """
{user_generator}

[Worse code]
{func_signature0}
{worse_code}

[Better code]
{func_signature1}
{better_code}

[Reflection]
{reflection}

[Improved code]
Please write an improved function `{func_name}`, according to the reflection. Output code only and enclose your code with Python code block: ```python ... ```."""

        self.mutation_prompt = """
{user_generator}

[Prior reflection]
{reflection}

[Code]
{func_signature1}
{elitist_code}

[Performance Feedback]
{elitist_performance}

[Improved code]
Please write a mutated function `{func_name}`, according to the reflection and performance feedback. Output code only and enclose your code with Python code block: ```python ... ```."""

        # Seed function
        self.seed_func = f"""def {self.func_name}(**kwargs):
    # your implementation here
    return # return required output format
"""

        self.seed_prompt = f"""
{self.seed_func}

Refer to the format of a trivial design above. Be very creative and give `{self.func_name}`. Output code only and enclose your code with Python code block: ```python ... ```."""

    def _init_population(self):
        """Initialize the population with seed function and random solutions."""
        # Evaluate the seed function
        logging.info("Initializing seed function...")
        code = extract_code_from_generator(self.seed_func)
        seed_ind = {
            "code": code,
            "response_id": 0,
            "exec_success": True,  # Assume it's valid
            "obj": float('-inf') if self.obj_type == "max" else float('inf'),  # Worst possible value
            "traceback": "",
            "detailed_feedback": "Baseline implementation"
        }
        self.seed_ind = seed_ind

        # Generate initial population
        logging.info("Generating initial population...")
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_reflection_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # Generate more diverse initial population with higher temperature
        responses = self.call_llm(
            messages,
            n=self.cfg.init_pop_size,
        )

        # Convert responses to individuals
        self.population = []
        for response_id, response in enumerate(responses):
            code = extract_code_from_generator(response)
            if code is not None:
                self.population.append({
                    "code": code,
                    "response_id": response_id,
                    "exec_success": True,  # Will be updated after evaluation
                    "obj": None,  # Will be updated after evaluation
                    "solution_id": self.solution_counter,
                    "traceback": "",
                    "detailed_feedback": ""
                })
                self.pending_solutions[self.solution_counter] = self.population[-1]
                self.solution_counter += 1

    def _rank_select(self) -> List[Dict]:
        """Select individuals with probability proportional to their rank."""
        # Filter out individuals without scores or failed execution
        valid_population = [ind for ind in self.population if ind["exec_success"] and ind["obj"] is not None]

        if len(valid_population) < 2:
            return None

        # Sort population by objective value (higher is better for max, lower for min)
        if self.obj_type == "max":
            valid_population = sorted(valid_population, key=lambda x: x["obj"], reverse=True)
        else:
            valid_population = sorted(valid_population, key=lambda x: x["obj"])

        ranks = list(range(len(valid_population)))
        probs = [1 / (rank + 1 + len(valid_population)) for rank in ranks]
        probs = [prob / sum(probs) for prob in probs]  # Normalize

        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(valid_population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None

        return selected_population

    def _short_term_reflection(self, ind1: Dict, ind2: Dict) -> Tuple[List, str, str]:
        """Generate short-term reflection for two individuals."""
        # Determine which individual is better
        if self.obj_type == "max":
            better_ind = ind1 if ind1["obj"] > ind2["obj"] else ind2
            worse_ind = ind2 if ind1["obj"] > ind2["obj"] else ind1
        else:
            better_ind = ind1 if ind1["obj"] < ind2["obj"] else ind2
            worse_ind = ind2 if ind1["obj"] < ind2["obj"] else ind1

        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        system = self.system_reflector_prompt
        user = f"""Below are two {self.func_name} functions for the following problem:
{self.problem_description}

You are provided with two code versions below, where the second version performs better than the first one.

[Worse code]
{worse_code}

[Better code]
{better_code}

[Worse code performance]
Score: {worse_ind["obj"]}
{worse_ind.get("detailed_feedback", "")}
{worse_ind.get("traceback", "")}

[Better code performance]
Score: {better_ind["obj"]}
{better_ind.get("detailed_feedback", "")}
{better_ind.get("traceback", "")}

You respond with some hints for designing better solutions, based on the two code versions and their performance feedback. Use less than 50 words."""

        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        return message, worse_code, better_code

    def _long_term_reflection(self, short_term_reflections: List[str]) -> None:
        """Generate long-term reflection based on short-term reflections."""
        system = self.system_reflector_prompt
        _short_term_reflections = "\n".join(short_term_reflections)
        user = f"""Below is your prior long-term reflection on designing solutions for the following problem:
{self.problem_description}

{self.long_term_reflection_str}

Below are some newly gained insights.
{_short_term_reflections}

Write constructive hints for designing better solutions, based on prior reflections and new insights and using less than 50 words."""

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        self.long_term_reflection_str = self.call_llm(messages)

        # Log the long-term reflection for transparency
        logging.info(f"Long-term reflection updated: {self.long_term_reflection_str}")

        # Write to file for persistence
        with open(f"{self.workspace}/long_term_reflection_iter_{self.iteration}.txt", 'w') as f:
            f.write(self.long_term_reflection_str)

    def _crossover(self, reflection: str, worse_code: str, better_code: str) -> Dict:
        """Create a new solution by crossover with reflection guidance."""
        system = self.system_generator_prompt
        func_signature0 = f"def {self.func_name}(**kwargs):"
        func_signature1 = f"def {self.func_name}(**kwargs):"

        user = self.crossover_prompt.format(
            user_generator=self.user_generator_prompt,
            func_signature0=func_signature0,
            func_signature1=func_signature1,
            worse_code=worse_code,
            better_code=better_code,
            reflection=reflection,
            func_name=self.func_name,
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = self.call_llm(messages)

        code = extract_code_from_generator(response)
        if code is not None:
            individual = {
                "code": code,
                "response_id": self.function_evals,
                "exec_success": True,  # Will be updated after evaluation
                "obj": None,  # Will be updated after evaluation
                "solution_id": self.solution_counter,
                "traceback": "",
                "detailed_feedback": "",
                "operation": "crossover",
                "reflections": reflection
            }
            self.function_evals += 1
            self.solution_counter += 1
            return individual
        return None

    def _mutate(self) -> Dict:
        """Create a new solution by mutating the elitist with reflection guidance."""
        if self.elitist is None:
            return None

        system = self.system_generator_prompt
        func_signature1 = f"def {self.func_name}(**kwargs):"

        # Include both the long-term reflection and performance details in the mutation
        elitist_performance = f"Score: {self.elitist['obj']}\n{self.elitist.get('detailed_feedback', '')}"

        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            reflection=self.long_term_reflection_str,
            func_signature1=func_signature1,
            elitist_code=filter_code(self.elitist["code"]),
            elitist_performance=elitist_performance,
            func_name=self.func_name,
        )

        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = self.call_llm(messages)

        code = extract_code_from_generator(response)
        if code is not None:
            individual = {
                "code": code,
                "response_id": self.function_evals,
                "exec_success": True,  # Will be updated after evaluation
                "obj": None,  # Will be updated after evaluation
                "solution_id": self.solution_counter,
                "traceback": "",
                "detailed_feedback": "",
                "operation": "mutation",
                "parent": self.elitist["solution_id"],
                "reflections": self.long_term_reflection_str
            }
            self.function_evals += 1
            self.solution_counter += 1
            return individual
        return None

    def _update_elitist(self) -> None:
        """Update the elitist individual."""
        valid_population = [ind for ind in self.population if ind["exec_success"] and ind["obj"] is not None]

        if not valid_population:
            return

        if self.obj_type == "max":
            best_obj = max(ind["obj"] for ind in valid_population)
            best_idx = next(i for i, ind in enumerate(valid_population) if ind["obj"] == best_obj)
        else:
            best_obj = min(ind["obj"] for ind in valid_population)
            best_idx = next(i for i, ind in enumerate(valid_population) if ind["obj"] == best_obj)

        best_individual = valid_population[best_idx]

        if self.elitist is None:
            self.elitist = best_individual
            logging.info(f"First elitist set with objective value: {best_obj}")
        else:
            if (self.obj_type == "max" and best_obj > self.elitist["obj"]) or \
                    (self.obj_type == "min" and best_obj < self.elitist["obj"]):
                prev_obj = self.elitist["obj"]
                self.elitist = best_individual
                logging.info(f"New elitist found with objective value: {best_obj} (improved from {prev_obj})")
                # Save the elitist code to a file for persistence
                with open(f"{self.workspace}/elitist_code_iter_{self.iteration}.py", 'w') as f:
                    f.write(best_individual["code"])

    def get_solution(self) -> Tuple[str, int]:
        """
        Get a new solution from the ReEvo agent.

        Returns:
            Tuple[str, int]: The solution code and a solution_id for scoring
        """
        # If we have pending solutions, return one of them
        if self.pending_solutions:
            solution_id = next(iter(self.pending_solutions))
            individual = self.pending_solutions[solution_id]
            return individual["code"], solution_id

        # Otherwise, we need to evolve new solutions
        self.iteration += 1
        logging.info(f"Starting iteration {self.iteration}")

        # If we have enough evaluated individuals
        if len([ind for ind in self.population if ind["exec_success"] and ind["obj"] is not None]) >= 2:
            # Select parents
            selected_population = self._rank_select()

            if selected_population is None or len(selected_population) < 2:
                # Fallback to generate a new solution if selection fails
                logging.info("Selection failed, generating a new solution")
                system = self.system_generator_prompt

                # Include long-term reflection if available
                reflection_prompt = ""
                if self.long_term_reflection_str:
                    reflection_prompt = f"\n\nBased on previous solutions, here are some tips to consider:\n{self.long_term_reflection_str}"

                user = self.user_generator_prompt + reflection_prompt
                messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
                response = self.call_llm(messages)
                code = extract_code_from_generator(response)
                individual = {
                    "code": code,
                    "response_id": self.function_evals,
                    "exec_success": True,
                    "obj": None,
                    "solution_id": self.solution_counter,
                    "traceback": "",
                    "detailed_feedback": "",
                    "operation": "fallback_new"
                }
                self.pending_solutions[self.solution_counter] = individual
                solution_id = self.solution_counter
                self.solution_counter += 1
                self.function_evals += 1
                return code, solution_id

            # Generate short-term reflections
            reflection_messages = []
            worse_codes = []
            better_codes = []

            for i in range(0, min(len(selected_population), 2 * self.cfg.pop_size), 2):
                if i + 1 >= len(selected_population):
                    break

                parent_1 = selected_population[i]
                parent_2 = selected_population[i + 1]

                message, worse_code, better_code = self._short_term_reflection(parent_1, parent_2)
                reflection_messages.append(message)
                worse_codes.append(worse_code)
                better_codes.append(better_code)

            # Get reflections
            reflections = []
            for message in reflection_messages:
                reflections.append(self.call_llm(message, model='gpt-4o-mini'))

            # Save short-term reflections for analysis
            for i, reflection in enumerate(reflections):
                with open(f"{self.workspace}/short_term_reflection_iter_{self.iteration}_{i}.txt", 'w') as f:
                    f.write(reflection)

            # Update long-term reflection
            self._long_term_reflection(reflections)

            # Select mutation or crossover based on probability
            if np.random.random() < self.cfg.mutation_rate and self.elitist is not None:
                # Mutation
                logging.info("Performing mutation")
                individual = self._mutate()
            else:
                # Crossover
                logging.info("Performing crossover")
                idx = np.random.randint(len(reflections))
                individual = self._crossover(
                    reflections[idx],
                    worse_codes[idx],
                    better_codes[idx]
                )

            if individual is not None:
                self.pending_solutions[individual["solution_id"]] = individual
                return individual["code"], individual["solution_id"]

        # If we don't have enough evaluated individuals or evolution failed, generate a new one
        logging.info("Generating a new solution (not enough evaluated individuals or evolution failed)")
        system = self.system_generator_prompt

        # Include long-term reflection if available
        reflection_prompt = ""
        if self.long_term_reflection_str:
            reflection_prompt = f"\n\nBased on previous solutions, here are some tips to consider:\n{self.long_term_reflection_str}"

        user = self.user_generator_prompt + reflection_prompt
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        response = self.call_llm(messages)
        code = extract_code_from_generator(response)
        individual = {
            "code": code,
            "response_id": self.function_evals,
            "exec_success": True,
            "obj": None,
            "solution_id": self.solution_counter,
            "traceback": "",
            "detailed_feedback": "",
            "operation": "new"
        }
        self.pending_solutions[self.solution_counter] = individual
        solution_id = self.solution_counter
        self.solution_counter += 1
        self.function_evals += 1
        return code, solution_id

    def receive_score(self, solution_id: int, score: float, detailed_score: Optional[str] = None) -> None:
        """
        Receive the evaluation score for a solution.

        Args:
            solution_id: The id of the solution to score
            score: The numerical score (higher is better for max objective)
            detailed_score: Optional detailed feedback about the solution's performance
        """
        if solution_id not in self.pending_solutions:
            logging.warning(f"Solution {solution_id} not found in pending solutions")
            return

        individual = self.pending_solutions[solution_id]

        # Adjust score based on objective type
        individual["obj"] = score if self.obj_type == "max" else -score
        individual["exec_success"] = True

        # Extract traceback if present in detailed_score
        if detailed_score:
            # traceback = filter_traceback(detailed_score)
            # if traceback:
            #     individual["traceback"] = traceback
            #     individual["exec_success"] = False  # Mark as failed if has traceback
            #     logging.info(f"Solution {solution_id} failed with traceback: {traceback[:100]}...")
            # Store detailed feedback
            individual["detailed_feedback"] = detailed_score

        # Add to population and remove from pending
        self.population.append(individual)
        del self.pending_solutions[solution_id]

        # Update elitist
        self._update_elitist()

        # Log metrics
        log_msg = f"Received score {score} for solution {solution_id}"
        if individual.get("operation"):
            log_msg += f" (operation: {individual['operation']})"

        valid_count = len([ind for ind in self.population if ind["exec_success"] and ind["obj"] is not None])
        log_msg += f" | Valid solutions: {valid_count}/{len(self.population)}"

        if self.elitist:
            log_msg += f" | Best score: {self.elitist['obj']}"

        logging.info(log_msg)



@dataclass
class Solution:
    code: str
    score: Optional[float] = None
    feedback: Optional[str] = None
    response: Optional[str] = None

class ReEvo:
    def __str__(self):
        return f"Greedy Refinement"

    def __init__(self, problem_description, timeout=10, model='openai/o3-mini', max_iter=64,
                 reasoning_effort='medium'):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = []
        self.solution_id = None
        self.reasoning_effort = reasoning_effort
        self.agent = ReEvoAgent(
            problem_description=problem_description,
            func_name="solve",
            obj_type="max",
            call_llm=partial(call_llm, model=model),
            population_size=10,
            init_population_size=4,
            mutation_rate=0.5,
            temperature=0.7
        )

    def step(self):
        solution_code, solution_id = self.agent.get_solution()
        self.solution_id = solution_id
        self.solution.append(Solution(code=solution_code, response=solution_code))
        return solution_code

    def feedback(self, score, feedback):
        self.agent.receive_score(self.solution_id, score, feedback)
        self.solution[-1].score = score
        self.solution[-1].feedback = feedback
        return

    def finalize(self):
        previous_best = sorted(self.solution, key=lambda x: x.score)[-1]
        return previous_best.code