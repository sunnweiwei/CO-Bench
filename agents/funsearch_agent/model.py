import numpy as np
import random
import time
from collections import defaultdict


class Solution:
    """Represents a solution to the optimization problem."""

    def __init__(self, code, prompt_id=None, version=None):
        self.code = code
        self.prompt_id = prompt_id
        self.version = version
        self.score = None
        self.score_detail = None

    def __len__(self):
        return len(self.code)


def softmax(x, temperature=1.0):
    """Compute softmax values for each set of scores in x."""
    e_x = np.exp((x - np.max(x)) / temperature)
    return e_x / e_x.sum()


class Cluster:
    """A cluster of solutions with the same score."""

    def __init__(self, score, solution):
        self.score = score
        self.solutions = [solution]

    def add_solution(self, solution):
        self.solutions.append(solution)

    def sample_solution(self):
        """Sample a solution, preferring shorter ones."""
        if len(self.solutions) == 1:
            return self.solutions[0]

        lengths = [len(s) for s in self.solutions]
        min_len = min(lengths)
        max_len = max(lengths)
        if min_len == max_len:
            return random.choice(self.solutions)

        normalized_lengths = [(l - min_len) / (max_len - min_len + 1e-6) for l in lengths]
        probs = softmax(-np.array(normalized_lengths))
        return self.solutions[np.random.choice(len(self.solutions), p=probs)]


class Island:
    """A subpopulation of solutions."""

    def __init__(self, solution_template, functions_per_prompt=2):
        self.solution_template = solution_template
        self.functions_per_prompt = functions_per_prompt
        self.clusters = {}  # score -> Cluster
        self.best_score = float('-inf')
        self.best_solution = None
        self.next_version = 0

    def register_solution(self, solution, score):
        """Register a solution with its score."""
        solution.score = score

        # Update best solution
        if score > self.best_score:
            self.best_score = score
            self.best_solution = solution

        # Add to appropriate cluster
        if score not in self.clusters:
            self.clusters[score] = Cluster(score, solution)
        else:
            self.clusters[score].add_solution(solution)

    def get_prompt(self):
        """Generate a prompt using the top solutions."""
        if not self.clusters:
            # Return the template if no solutions yet
            return self.solution_template, self.next_version

        # Choose clusters based on score
        scores = list(self.clusters.keys())
        scores.sort(reverse=True)  # Sort in descending order

        # Take the top N scores
        num_solutions = min(self.functions_per_prompt, len(scores))
        chosen_scores = scores[:num_solutions]

        # Sample solutions from each chosen cluster
        chosen_solutions = [self.clusters[score].sample_solution() for score in chosen_scores]

        # Generate prompt with previous solutions
        prompt = self._generate_prompt(chosen_solutions)
        version = self.next_version
        self.next_version += 1
        return prompt, version

    def _generate_prompt(self, solutions):
        """Create a prompt that incorporates the previous solutions."""
        prompt = self.solution_template + "\n\n"
        prompt += ("# Here are some previous solutions for reference. "
                   "Note that the score is normalized relative to the reference score, "
                   "where a higher value is always better "
                   "(score 1.0 mean the performance is same as the reference score):\n\n")

        for i, solution in enumerate(solutions):
            prompt += f"# Solution {i + 1} (score: {solution.score}):\n"
            prompt += solution.code + "\n\n"
            prompt += "Per instances score:" + solution.score_detail + "\n\n"

        prompt += ("# Please provide an improved solution that addresses the limitations of previous attempts. "
                   "You can analyse above evaluation results and think about how to improve it."
                   "For this you should first outline a brief plan in natural language for how the solution can be improved and "
                   "then implement this improvement in Python based on the provided previous solution."
                   "Ensure your algorithm is as effective as possible. You may use any Python package. "
                   "Your new solution should be significantly different and better than previous solution. "
                   "Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs) "
                   "Do not use -----, make sure use ```python ... ``` to enclose your code. "
                   "Your function has timeout; aim to return the best possible results within this limit.")
        return prompt


class Funsearch:
    """Implementation of the Funsearch methodology for operations research problems."""

    def __init__(self, problem_description, num_islands=10, functions_per_prompt=2,
                 reset_period=4 * 60 * 60, timeout=10):
        """Initialize the Funsearch system.

        Args:
            problem_description: Description of the optimization problem
            solution_template: Template for the solution function
            num_islands: Number of islands to maintain for diversity
            functions_per_prompt: Number of previous solutions to include in prompts
            reset_period: How often (in seconds) to reset weaker islands
        """
        self.problem_description = problem_description

        # Add problem description to the template
        full_template = (f"You are an expert in Operation Research problem. Solve the following problem:\n\n"
                         f"# Problem Description:\n{problem_description}\n\n"
                         f"Ensure your algorithm is as effective as possible. You may use any Python package. "
                         f"Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs)`. "
                         f"Your function has a {timeout}-second timeout; aim to return the best possible results within this limit.")

        # Initialize islands
        self.num_islands = num_islands
        self.islands = [Island(full_template, functions_per_prompt) for _ in range(num_islands)]

        # Track prompt information
        self.prompts = {}  # prompt_id -> (island_id, version, solution)
        self.next_prompt_id = 0

        # Reset parameters
        self.reset_period = reset_period
        self.last_reset_time = time.time()

        # Best solution tracking
        self.best_scores = [float('-inf')] * num_islands
        self.best_solutions = [None] * num_islands

    def get_prompt(self):
        """Get the next prompt to send to the LLM."""
        # Choose an island randomly
        island_id = random.randint(0, self.num_islands - 1)
        island = self.islands[island_id]

        # Get prompt from the island
        prompt, version = island.get_prompt()

        # Store the prompt details
        prompt_id = self.next_prompt_id
        self.prompts[prompt_id] = (island_id, version, None)
        self.next_prompt_id += 1

        return prompt, prompt_id

    def pull_score(self, prompt_id, score, score_detail, solution_code):
        """Process a score for a generated solution.

        Args:
            prompt_id: ID of the prompt that generated the solution
            score: Evaluation score of the solution
            solution_code: The solution code generated by the LLM
        """
        if prompt_id not in self.prompts:
            raise ValueError(f"Unknown prompt ID: {prompt_id}")

        island_id, version, _ = self.prompts[prompt_id]

        # Create a solution object
        solution = Solution(solution_code, prompt_id, version)
        solution.score = score
        solution.score_detail = score_detail

        # Update the prompts dictionary with the solution
        self.prompts[prompt_id] = (island_id, version, solution)

        # Register the solution with the appropriate island
        self.islands[island_id].register_solution(solution, score)

        # Check if this solution improves the best solution for this island
        if score > self.best_scores[island_id]:
            self.best_scores[island_id] = score
            self.best_solutions[island_id] = solution

        # Check if it's time to reset islands
        current_time = time.time()
        if current_time - self.last_reset_time > self.reset_period:
            self.reset_islands()
            self.last_reset_time = current_time

    def reset_islands(self):
        """Reset the weaker islands to maintain diversity."""
        # Sort islands by their best score
        indices = np.argsort(self.best_scores)

        # Reset the bottom half of islands
        num_to_reset = self.num_islands // 2
        for i in range(num_to_reset):
            island_id = indices[i]

            # Choose a donor island from the top half
            donor_id = indices[-(i % (self.num_islands - num_to_reset) + 1)]

            # Create a new island with the same template
            self.islands[island_id] = Island(self.islands[island_id].solution_template,
                                             self.islands[island_id].functions_per_prompt)

            # Seed it with the best solution from the donor island
            if self.best_solutions[donor_id] is not None:
                donor_solution = self.best_solutions[donor_id]
                self.islands[island_id].register_solution(donor_solution, donor_solution.score)

            # Reset the best score for this island
            self.best_scores[island_id] = self.islands[island_id].best_score
            self.best_solutions[island_id] = self.islands[island_id].best_solution