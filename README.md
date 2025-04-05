# CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization


![example](https://github.com/user-attachments/assets/faf29c44-4904-4d74-9a15-37a038b14e77)

**Paper:** [CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization](preprint.pdf)

**Data:** [CO-Bench](https://huggingface.co/datasets/CO-Bench/CO-Bench)

# Download Data
Download the raw data from [https://huggingface.co/datasets/CO-Bench/CO-Bench](https://huggingface.co/datasets/CO-Bench/CO-Bench) to the local directory `data`
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='CO-Bench/CO-Bench',
    repo_type='dataset',
    local_dir='data'
)
```


# Evaluation
Below is code to run evaluation of *Greedy Refinement* agent on `Aircraft landing` for 64 iterations.
```python
from agents import GreedyRefine, DirectAnswer, FunSearch, AIDE
from evaluation.evaluate import Evaluator
from evaluation.controller import get_data

# Load data
data = get_data('Aircraft landing', src_dir='data')

# Define agent
agent = GreedyRefine(
    problem_description=data.problem_description,
    timeout=10,
    model='openai/o3-mini'
)

# Load evaluator
evaluator = Evaluator(data)

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
from agents import GreedyRefine, DirectAnswer, FunSearch, AIDE
agent = GreedyRefine(
    problem_description=data.problem_description,
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
    agent.feedback(feedback.dev_score, feedback.dev_feedback) 

# Get the final soltuion
code = agent.finalize()
print(code)
```
</details>

# Cite

```
@article{Sun2023COBench,
  title={CO-Bench: Benchmarking Language Model Agents in Algorithm Search for Combinatorial Optimization},
  author={Weiwei Sun and Shengyu Feng and Shanda Li and Yiming Yang},
  journal={ArXiv},
}
```
