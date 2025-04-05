from agents.utils import call_llm, extract_code_blocks, textwrap


class DirectAnswer:
    def __str__(self):
        return f"Directly Answer"

    def __init__(self, problem_description, timeout=10, model='openai/o3-mini', reasoning_effort='medium', **kwargs):
        self.problem_description = problem_description
        self.timeout = timeout
        self.model = model
        self.solution = None
        self.reasoning_effort = reasoning_effort

    def step(self):
        if self.solution is not None:
            return None
        prompt = (
            f"You are an expert in Operation Research problem. "
            f"Solve the following problem:\n\n{self.problem_description}\n\n"
            f"Ensure your algorithm is as effective as possible. You may use any Python package. "
            f"Enclose all your code within a code block: ```python ... ``` and name the main function `def solve(**kwargs)`. "
            f"Your function has a {self.timeout}-second timeout; aim to return the best possible results within this limit."
        )
        response = call_llm(prompt, model=self.model, reasoning_effort=self.reasoning_effort)
        code_blocks = extract_code_blocks(response)
        code = textwrap.dedent(code_blocks[0])
        self.solution = code
        return code

    def feedback(self, score, feedback, ):
        return

    def finalize(self):
        return self.solution


def main():
    src_dir = 'data_new2'
    all_dir = ['Aircraft landing',
               # 'Assignment problem',
               'Assortment problem', 'Bin packing - one-dimensional',
               'Capacitated warehouse location', 'Common due date scheduling', 'Constrained guillotine cutting',
               'Constrained non-guillotine cutting', 'Container loading', 'Container loading with weight restrictions',
               'Corporate structuring', 'Crew scheduling', 'Equitable partitioning problem',
               'Euclidean Steiner problem', 'Flow shop scheduling', 'Generalised assignment problem', 'Graph colouring',
               'Hybrid Reentrant Shop Scheduling', 'Job shop scheduling', 'MIS',
               'Multi-Demand Multidimensional Knapsack problem', 'Multidimensional knapsack problem',
               'Open shop scheduling', 'Packing unequal circles', 'Packing unequal circles area',
               'Packing unequal rectangles and squares', 'Packing unequal rectangles and squares area',
               'Resource constrained shortest path', 'Set covering', 'Set partitioning', 'TSP',
               'Uncapacitated warehouse location', 'Unconstrained guillotine cutting',
               'Vehicle routing: period routing', 'p-median - capacitated', 'p-median - uncapacitated']
    print(all_dir)

    timeout = 10

    # model_name = 'gpt-4o-mini'
    # model_name = 'gpt-4o'
    # model_name = 'o3-mini'
    # model_name = 'gemini-2.5-pro'
    # model_name = 'o1-high'
    # model_name = 'claude-37-sonnet-thinking'
    # model_name = 'DeepSeek-V3'
    # model_name = 'QwQ-32B'
    # model_name = 'Qwen2.5-Coder-32B-Instruct'
    # model_name = 'Llama-3.3-70B-Instruct'
    # model_name = 'gemini-2.0-flash-thinking'
    model_name = 'grok-3-thinking'

    name_tag = 'direct'
    # name_tag = 'gurobi'

    # name_tag = 'chain-v1'

    # name_tag = 'bon-v2'

    # name_tag = 'aide-v1'

    # old_tag = 'bon-v2'
    # old_tag = name_tag

    # name_tag = 'fun-v2'
    # old_tag = 'fun-v1'

    cpu_num = 30

    # call_func = call_llm
    # call_func = call_together
    # call_func = 'claude'
    call_func = 'grok'
    # call_func = 'solution'
    # call_func = 'xx'

    print(model_name)

    task_id = 0
    for task in all_dir:
        if task in ['Steiner problem in graphs', 'Boxes on shelves', 'Index tracking', 'Portfolio rebalancing',
                    'Period travelling salesman', 'Vehicle routing: fixed areas', 'Portfolio optimisation']:
            continue

        # if task not in ['Resource constrained shortest path']:
        #     continue

        # if task not in ['Open shop scheduling']:
        #     continue

        print('#' * 50)
        print(task_id, task)
        print('#' * 50)
        #
        if os.path.exists(f"Score/{task}/{name_tag}_{model_name}.txt"):
            print(f"exists: Score/{task}/{name_tag}_{model_name}.txt")
            record = json.load(open(f"Score/{task}/{name_tag}_{model_name}.txt"))
            if "'float' object cannot be interpreted as an integer" in str(record['results']):
                print('But have data stype error, retry!')
                continue
            elif "No module named " in str(record['results']):
                print('But have module error, retry!')
                continue
            elif "gurobi" in str(record['results']).lower():
                print(record['results'])
                print("Gurobi error, retry")
            elif 'ortools' in str(record['results']).lower():
                print(record['results'])
                print("ortools error, retry")
            else:
                # print(record['results'])
                continue

        try:
            load_data, _, problem = import_func(f"{src_dir}/{task}/config.py", 'load_data', 'eval_func', 'DESCRIPTION')
        except Exception as e:
            print(f'Error when loading {task}', e)
            continue

        task_id += 1
        config_path = f"{src_dir}/{task}/config.py"
        solve_template = extract_function_source(f"{src_dir}/{task}/config.py", 'solve')
        test_cases = list_test_cases(f"{src_dir}/{task}")

        solve_template = f"{problem}\n\n## **Implement in Solve Function**\n\n{solve_template}"
        try:
            norm_score, = import_func(f"{src_dir}/{task}/config.py", 'norm_score')
        except AttributeError:
            norm_score = lambda x: x

        all_results = []

        # name_tag = 'bon'

        agent = AgentPipeline()
        prompt = agent.draft(solve_template, timeout=timeout)
        # prompt += ' do not use ortools.'
        # with open('prompt.txt', 'w') as f:
        #     f.write(prompt)
        # print('Input Results:')
        # _ = input('>>')
        # with open('response.txt', 'r') as f:
        #     response = f.read()

        if call_func == 'claude':
            response = open(f"Bridge/{task}/{name_tag}_{model_name}.txt").read()
            solve_source = response
        elif call_func == 'grok':
            response = open(f"Bridge/{task}/{name_tag}_{model_name}.txt").read()
            code_blocks = extract_code_blocks(response)
            solve_source = textwrap.dedent(code_blocks[0])
        elif call_func == 'solution':
            response = open(f"Solution/{task}/{name_tag}_{model_name}.py").read()
            solve_source = response
        elif isinstance(call_func, str):
            response = open(f"Response/{task}/{name_tag}_{model_name}.txt").read()
            code_blocks = extract_code_blocks(response)
            solve_source = textwrap.dedent(code_blocks[0])
        else:
            response = call_func(prompt, model=model_name)
            if '</think>' in response:
                code_blocks = extract_code_blocks(response[response.rindex('</think>'):])
            else:
                code_blocks = extract_code_blocks(response)
            solve_source = textwrap.dedent(code_blocks[0])

        # print(response)
        # response = open(f"Response/{task}/{name_tag}_{model_name}.txt").read()
        # code_blocks = extract_code_blocks(response)

        # solve_source = textwrap.dedent(code_blocks[0])
        #
        # response = open(f"Bridge/{task}/{name_tag}_{model_name}.txt").read()
        # solve_source = response

        price = CostTracker.total_cost_usd

        data_size = {case: [1] * len(load_data(f"{src_dir}/{task}/{case}")) for case in test_cases}
        case_workers, instance_workers = design_optimal(data_size, cpu_num)
        # print(case_workers, instance_workers)
        results = process_all_cases(test_cases, task, load_data, solve_source, config_path, src_dir,
                                    timeout=timeout, instance_workers=instance_workers, case_workers=case_workers)
        # print(results)
        results = norm_score(results)
        for case in test_cases:
            scores, error_message = results.get(case, (None, "No result"))
            if error_message:
                print(f"{case} -> Caught Error: {error_message}")
            else:
                print(f"{case} -> Scores: {scores}")

        score = eval_all(results, test_cases)
        print(model_name)
        print(f'# Avg Score:\n{score}\n')

        all_results.append({'response': response, 'score': score, 'price': price,
                            'results': results, 'task': task, 'prompt': prompt})

        # os.makedirs(f"Solution/{task}", exist_ok=True)
        # os.makedirs(f"Response/{task}", exist_ok=True)
        # os.makedirs(f"out/{task}", exist_ok=True)
        os.makedirs(f"Score/{task}", exist_ok=True)
        # with open(f"Solution/{task}/{name_tag}_{model_name}.py", "w") as file:
        #     file.write(solve_source)
        # with open(f"Response/{task}/{name_tag}_{model_name}.txt", "w") as file:
        #     file.write(response)
        # with open(f'out/{task}/{name_tag}_{model_name}.json', 'w') as f:
        #     json.dump(all_results, f)
        with open(f"Score/{task}/{name_tag}_{model_name}.txt", "w") as f:
            json.dump({'task': task, 'price': price, 'score': score, 'results': results}, f)
        # os.system('pkill -u weiweis -f pulp')


if __name__ == '__main__':
    main()
