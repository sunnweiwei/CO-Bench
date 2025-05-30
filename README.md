# CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization


![example](https://github.com/user-attachments/assets/faf29c44-4904-4d74-9a15-37a038b14e77)

**Paper:** [CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization](https://arxiv.org/abs/2504.04310)

**Data:** [CO-Bench](https://huggingface.co/datasets/CO-Bench/CO-Bench)

# News

- **[2025.5.24]** Check out [FrontierCO](https://arxiv.org/abs/2505.16952), a benchmark of real-world, challenging (and in some cases unsolved) combinatorial optimization problems. See for [Evaluation on FrontierCO](#evaluation-on-frontierco) section for code to evaluate agents on this benchmark.
- **[2025.4.6]** Released CO-Bench.

# Download Data
Download the raw data from [https://huggingface.co/datasets/CO-Bench/CO-Bench](https://huggingface.co/datasets/CO-Bench/CO-Bench) to the local directory `data`

```
huggingface-cli download CO-Bench/CO-Bench --repo-type dataset --local-dir data
```
or

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='CO-Bench/CO-Bench',
    repo_type='dataset',
    local_dir='data'
)
```

# Agent Implementations

Agents are implemented in the `agents` module. Currently supported agents include: `GreedyRefine`, `DirectAnswer`, `BestOfN`, `FunSearch` ([link](https://github.com/google-deepmind/funsearch)), `AIDE` ([link](https://github.com/WecoAI/aideml)), `ChainOfExperts` ([link](https://github.com/xzymustbexzy/Chain-of-Experts)), `ReEvo` ([link](https://github.com/ai4co/reevo)), EoH ([link](https://github.com/FeiLiu36/EoH)), MCTS-AHD ([link](https://github.com/zz1358m/MCTS-AHD-master)). 
LLMs are supported via [liteLLM](https://github.com/BerriAI/litellm).

Each agent implements the following functions:
- `step()`: Returns the next candidate code for evaluation.
- `feedback()`: Accepts evaluation results of previous candidate code.
- `finalize()`: Returns the final code.

# Evaluation
![image](https://github.com/user-attachments/assets/b1206bfb-711e-4f4b-ab11-c096df4286cd)
Below is code to run evaluation of *Greedy Refinement* agent on `Aircraft landing` for 64 iterations.
```python
from agents import GreedyRefine, DirectAnswer, FunSearch, AIDE, ChainOfExperts, ReEvo, BestOfN
from evaluation import Evaluator, get_data

# Load data
data = get_data('Aircraft landing', src_dir='data')

# Define agent, here we use GreedyRefine
agent = GreedyRefine(
    problem_description=data.problem_description,
    timeout=10,
    model='openai/o3-mini', # We use LiteLLM to call API
)

# Load evaluator
evaluator = Evaluator(data, timeout=10)

# Run for 64 iterations
for it in range(64):
    code = agent.step()
    if code is None:  # agent decides to terminate
        break
    feedback = evaluator.evaluate(code)  # Run evaluation
    agent.feedback(feedback.dev_score, feedback.dev_feedback)  # Use dev set score as feedback

# Get the final solution
code = agent.finalize()
feedback = evaluator.evaluate(code)
print(feedback.test_feedback)  # Test set score
```


**Evaluation on All Tasks**

```python
TASK_LIST = ['Aircraft landing', 'Assignment problem', 'Assortment problem', 'Bin packing - one-dimensional', 'Capacitated warehouse location', 'Common due date scheduling', 'Constrained guillotine cutting', 'Constrained non-guillotine cutting', 'Container loading', 'Container loading with weight restrictions', 'Corporate structuring', 'Crew scheduling', 'Equitable partitioning problem', 'Euclidean Steiner problem', 'Flow shop scheduling', 'Generalised assignment problem', 'Graph colouring', 'Hybrid Reentrant Shop Scheduling', 'Job shop scheduling', 'MIS', 'Multi-Demand Multidimensional Knapsack problem', 'Multidimensional knapsack problem', 'Open shop scheduling', 'Packing unequal circles', 'Packing unequal circles area', 'Packing unequal rectangles and squares', 'Packing unequal rectangles and squares area', 'Resource constrained shortest path', 'Set covering', 'Set partitioning', 'TSP', 'Uncapacitated warehouse location', 'Unconstrained guillotine cutting', 'Vehicle routing: period routing', 'p-median - capacitated', 'p-median - uncapacitated']

for task in TASK_LIST:
    ... # Run evaluation on one task
```

<details>
<summary><strong>Using Agents on Custom Problems</strong></summary>

Step 1: Include a concise description and a solve template. For example:

```python
problem_description = '''The Traveling Salesman Problem (TSP) is a classic combinatorial optimization problem where, given a set of cities with known pairwise distances, the objective is to find the shortest possible tour that visits each city exactly once and returns to the starting city. More formally, given a complete graph G = (V, E) with vertices V representing cities and edges E with weights representing distances, we seek to find a Hamiltonian cycle (a closed path visiting each vertex exactly once) of minimum total weight.

Implement in Solve Function

def solve(**kwargs):
    """
    Solve a TSP instance.

    Args:
        - nodes (list): List of (x, y) coordinates representing cities in the TSP problem
                      Format: [(x1, y1), (x2, y2), ..., (xn, yn)]

    Returns:
        dict: Solution information with:
            - 'tour' (list): List of node indices representing the solution path
                           Format: [0, 3, 1, ...] where numbers are indices into the nodes list
    """

    return {
        'tour': [],
    }
'''
```
Step 2: Define the agent
```python
from agents import GreedyRefine
agent = GreedyRefine(
    problem_description=problem_description,
    timeout=10,
    model='openai/o3-mini')
```
Step 3: Define the `evaluate` function and run the loop. Use the evaluate function to get results on the data, and iteratively improve the solution based on feedback:
```python
evaluate = ...  # Define evaluate() to return score (float) and feedback (str)
# Run for 64 iterations
for it in range(64):
    code = agent.step()
    dev_score, dev_feedback = evaluate(code) # Define evaluate() to return score (float) and feedback (str)
    agent.feedback(dev_score, dev_feedback) 

# Get the final soltuion
code = agent.finalize()
print(code)
```
</details>

*Docker environment for sandboxed agent execution and solution evaluation: coming soon.*

# Evaluation on FrontierCO

Download the raw data from [https://huggingface.co/datasets/CO-Bench/FrontierCO](https://huggingface.co/datasets/CO-Bench/FrontierCO) to the local directory `data`:

```
huggingface-cli download CO-Bench/FrontierCO --repo-type dataset --local-dir data
```
or

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='CO-Bench/FrontierCO',
    repo_type='dataset',
    local_dir='data'
)
```

Note: In FrontierCO, instead of returning a single solution, the solve function must **yield increasingly better solutions** over time. The solve template looks like this:

```python
def solve(**kwargs):
    """
    Solve a TSP instance.

    Args:
        - nodes (list): List of (x, y) coordinates representing cities in the TSP problem.
                        Format: [(x1, y1), (x2, y2), ..., (xn, yn)]

    Yields:
        dict: Solution information with:
            - 'tour' (list): List of node indices representing the solution path.
                             Format: [0, 3, 1, ...] where numbers are indices into the nodes list.
    """
    # Your function must yield multiple solutions over time, not just return one.
    # Use Python's yield keyword repeatedly to produce a stream of solutions.
    # Each yielded solution should be better than the previous one.
    # Evaluation is based on the last yielded solution before timeout.
    while True:
        yield {
            'tour': [],
        }
```

Compared with CO-Bench, we use new agent implementations in FrontierCO:

```python
from agents import YieldGreedyRefine, YieldFunSearch, YieldReEvo

# And a new evaluator to fetch solutions yielded by the solver,
# evaluating only the last solution before timeout:
from evaluation import YieldingEvaluator, get_new_data

# Load data
data = get_new_data(task, src_dir='data', data_dir='data')

# Define agent (example: YieldGreedyRefine)
agent = YieldGreedyRefine(
    problem_description=data.problem_description,
    timeout=300,  # 300s timeout during solver development
    model='openai/o3-mini',  # We use LiteLLM to call the API
)

# Load YieldingEvaluator
# 300s timeout during solver development
evaluator = YieldingEvaluator(data, timeout=300)

# Run for 64 iterations
for it in range(64):
    code = agent.step()
    if code is None:  # agent decides to terminate
        break
    feedback = evaluator.evaluate(code)  # Run evaluation
    agent.feedback(feedback.dev_score, feedback.dev_feedback)  # Use dev set score as feedback

# Get the final solution
code = agent.finalize()

# For final evaluation, run the solver for 1 hour
final_evaluator = YieldingEvaluator(data, timeout=60 * 60)
feedback = final_evaluator.evaluate(code)
print(feedback.test_feedback)  # Test set score
```


# Cite
```
@article{Sun2025COBench,
  title={CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization},
  author={Weiwei Sun and Shengyu Feng and Shanda Li and Yiming Yang},
  journal={ArXiv},
  year={2025},
  volume={abs/2504.04310},
  url={https://arxiv.org/abs/2504.04310},
}
```

```
@article{Feng2025FrontierCO,
  title={A Comprehensive Evaluation of Contemporary ML-Based Solvers for Combinatorial Optimization},
  author={Shengyu Feng and Weiwei Sun and Shanda Li and Ameet Talwalkar and Yiming Yang},
  journal={ArXiv},
  year={2025},
  volume={abs/2505.16952},
  url={https://arxiv.org/abs/2505.16952},
}
```
