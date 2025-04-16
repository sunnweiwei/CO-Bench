from agents.chain_expert.utils import extract_code_from_string, call_llm


def solve(problem, model_name='gpt-3.5-turbo'):
    prompt_template = """You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.

You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem}

Let's solve this problem step by step:

1. First, let's identify the decision variables in this optimization problem.
2. Next, let's define the objective function that we want to optimize.
3. After that, let's identify and formulate the constraints.
4. Finally, let's put everything together and solve the model.

Give your Python code, with detailed comments explaining your approach and each step of the solution."""
    
    prompt = prompt_template.replace("{problem}", problem)
    answer = call_llm(model_name, prompt)
    code = extract_code_from_string(answer)
    return code
