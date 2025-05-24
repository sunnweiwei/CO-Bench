import os
import math
import importlib.util
import openai
import re
import textwrap
import signal
import sys
import io
import concurrent.futures
import ast
import contextlib
from tqdm import tqdm
import subprocess
import multiprocessing as mp
import json
import psutil
import signal
import subprocess
import os
import signal
import multiprocessing as mp
import fcntl

import cloudpickle
from multiprocessing.reduction import ForkingPickler

# Use cloudpickle to support pickling dynamic functions.
ForkingPickler.dumps = cloudpickle.dumps


class TimeoutException(Exception):
    pass


def timeout_handler(signum, frame):
    raise TimeoutException("Execution time exceeded 60 seconds")


def read_file(path):
    return "".join([line for line in open(path)])


def write_to_file(filename: str, content: str):
    with open(filename, 'w', encoding='utf-8') as file:
        file.write(content)


def import_func(path, *var_names):
    # Use the filename (without extension) as the module name.
    module_name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return (getattr(module, var) for var in var_names)


def read_eval_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
        return content
    except FileNotFoundError:
        return f"File not found: {file_path}"
    except Exception as e:
        return f"An error occurred while reading the file: {e}"


def list_dirs(path="."):
    return sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])


def list_test_cases(path="."):
    return sorted(
        f for f in os.listdir(path)
        if not (f.endswith(".py") or f == "__pycache__")
    )


class FileLock:
    def __init__(self, lock_file_path='cpu.lock'):
        self.lock_file_path = lock_file_path
        self.lock_file = None

    def __enter__(self):
        # Open (or create) the lock file
        self.lock_file = open(self.lock_file_path, "w")
        # Acquire an exclusive lock (this will block until the lock is available)
        fcntl.flock(self.lock_file, fcntl.LOCK_EX)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # Release the lock and close the file
        fcntl.flock(self.lock_file, fcntl.LOCK_UN)
        self.lock_file.close()

class CostTracker:
    total_cost_usd = 0.0

    @classmethod
    def add_cost(cls, cost: float):
        cls.total_cost_usd += cost

    @classmethod
    def get_total_cost(cls) -> float:
        return cls.total_cost_usd



def call_llm(question: str, model='openai/gpt-4o', reasoning_effort=None) -> str:
    from litellm import completion
    messages = [{"content": question, "role": "user"}]
    response = completion(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.choices[0].message.content


def extract_and_compile_code(llm_answer: str):
    # This function is still useful for testing in the main process if needed.
    code_blocks = re.findall(r"```python(.*?)```", llm_answer, re.DOTALL)
    if not code_blocks:
        raise ValueError("No Python code block found in the LLM response.")
    extracted_code = textwrap.dedent(code_blocks[0])
    if "def solve(" not in extracted_code:
        raise ValueError("Extracted code does not define a function named 'solve'.")
    namespace = {}
    try:
        exec(extracted_code, namespace)
    except Exception as e:
        raise RuntimeError(f"Error executing the extracted code: {e}")
    if "solve" not in namespace or not callable(namespace["solve"]):
        raise ValueError("Extracted code does not contain a valid 'solve' function.")
    return namespace["solve"]


def extract_function_source(file_path: str, function_name: str) -> str:
    with open(file_path, 'r', encoding='utf-8') as f:
        source = f.read()
    tree = ast.parse(source, filename=file_path)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == function_name:
            start_line = node.lineno - 1
            if not hasattr(node, 'end_lineno'):
                raise RuntimeError("Python 3.8+ is required for this function to work properly.")
            end_line = node.end_lineno
            source_lines = source.splitlines()
            function_source = "\n".join(source_lines[start_line:end_line])
            return function_source
    raise ValueError(f"Function '{function_name}' not found in the file '{file_path}'.")


def design_optimal(problem_cases, K):
    def simulate(N, M):
        slots = [0] * N
        for cases in problem_cases.values():
            t = math.ceil(len(cases) / M)
            slots[slots.index(min(slots))] += t
        return max(slots)

    best_time, best_N, best_M = float('inf'), None, None
    P = len(problem_cases)

    for N in range(1, P + 1):
        M = K // N
        if M < 1:
            continue
        total_time = simulate(N, M)
        # Prefer smaller N if total_time is the same
        if total_time < best_time or (total_time == best_time and N < best_N):
            best_time, best_N, best_M = total_time, N, M

    return best_N, best_M

@contextlib.contextmanager
def capture_all_output():
    buffer = io.StringIO()
    # Save the original stdout and stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout, sys.stderr = buffer, buffer
    # For subprocess calls that expect file descriptors, we may need to use the actual file descriptor
    stdout_fd = old_stdout.fileno()
    stderr_fd = old_stderr.fileno()
    saved_stdout_fd = os.dup(stdout_fd)
    saved_stderr_fd = os.dup(stderr_fd)
    try:
        yield buffer
    finally:
        # Restore original stdout and stderr
        sys.stdout, sys.stderr = old_stdout, old_stderr
        os.dup2(saved_stdout_fd, stdout_fd)
        os.dup2(saved_stderr_fd, stderr_fd)
        os.close(saved_stdout_fd)
        os.close(saved_stderr_fd)



class ParallelRun:
    def __init__(self, func, *args, **kwargs):
        self.func = func

    def evaluate_instance_in_subprocess(self, instance, solve_source, config_path, queue):
        """
        Run evaluation inside a process and store its PID in a global variable
        so we can identify its children later if needed.
        """
        try:
            # Set process group ID to make it easier to kill all children later
            if hasattr(os, 'setpgrp'):  # Unix/Linux/Mac
                os.setpgrp()

            # Re-import eval_func from the config file.
            _, eval_func = import_func(config_path, 'load_data', 'eval_func')
            # Compile the solve function from its source code.
            local_namespace = {}
            exec(solve_source, local_namespace)
            if "solve" not in local_namespace:
                raise ValueError("The source code does not define a 'solve' function.")
            solve_func = local_namespace["solve"]
            # result = evaluate_instance(instance, solve_func, eval_func)

            with capture_all_output():
                result = self.func(instance, solve_func, eval_func)
            queue.put(result)
        except Exception as e:
            queue.put(f"Exception: {str(e)}")



    def run_instance_with_timeout(self, instance, solve_source, config_path, timeout):
        # Create a unique cgroup name for this instance.
        # (You might use a unique identifier from the instance or the process PID)
        cgroup_name = f"experiment_{os.getpid()}_{instance.get('id', 'unknown')}"

        # Create a cgroup for CPU and memory (adjust as needed for your system, and note this works for cgroup v1)
        # subprocess.run(["cgcreate", "-g", f"cpu,memory:/{cgroup_name}"],
        #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        queue = mp.Queue()
        p = mp.Process(target=self.evaluate_instance_in_subprocess,
                       args=(instance, solve_source, config_path, queue))
        p.start()

        # Add the process to the cgroup
        # subprocess.run(["cgclassify", "-g", f"cpu,memory:/{cgroup_name}", str(p.pid)],
        #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        p.join(timeout + 1)  # 1 extra second
        if p.is_alive():
            p.terminate()
            try:
                parent = psutil.Process(p.pid)
                it = 1
                for child in parent.children(recursive=True):
                    if it > 100:
                        break
                    child.kill()
                    it += 1
                parent.kill()
            except psutil.NoSuchProcess:
                pass
            p.join(1)
            # Kill all processes in the cgroup (including detached pulp solvers)
            # subprocess.run(["cgdelete", "-g", f"cpu,memory:/{cgroup_name}"],
            #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return f"Timeout ({timeout}s)"
        else:
            try:
                result = queue.get_nowait()
            except Exception:
                result = "No result"
            # subprocess.run(["cgdelete", "-g", f"cpu,memory:/{cgroup_name}"],
            #                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return result


    def process_single_case(self, case, task, load_data, solve_source, config_path, src_dir, timeout, instance_workers):
        # print(case)
        file_path = os.path.join(src_dir, task, case)
        list_of_instance = load_data(file_path)
        # print('Load done')
        list_of_instance = list_of_instance
        inst_total = len(list_of_instance)
        instance_results = [None] * inst_total

        with concurrent.futures.ThreadPoolExecutor(max_workers=instance_workers) as instance_executor:
            future_to_idx = {
                instance_executor.submit(self.run_instance_with_timeout, instance, solve_source, config_path, timeout): idx
                for idx, instance in enumerate(list_of_instance)
            }
            for future in concurrent.futures.as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    result = future.result()
                except Exception as e:
                    result = f"Exception: {str(e)}"
                instance_results[idx] = result

        return case, (instance_results, None)


    def process_all_cases(self, test_cases, task, load_data, solve_source, config_path, src_dir,
                          timeout=60, instance_workers=4, case_workers=4):
        results = {}
        pbar = tqdm(total=len(test_cases), desc=f"Processing cases for '{task}'", unit="case")

        # Submit each case processing as an independent process.
        with concurrent.futures.ProcessPoolExecutor(max_workers=case_workers) as case_executor:
            future_to_case = {
                case_executor.submit(
                    self.process_single_case, case, task, load_data, solve_source, config_path, src_dir, timeout,
                    instance_workers
                ): case for case in test_cases
            }
            for future in concurrent.futures.as_completed(future_to_case):
                try:
                    case, case_result = future.result()
                except Exception as e:
                    case = future_to_case[future]
                    case_result = (None, f"Exception: {str(e)}")
                results[case] = case_result
                pbar.update(1)
        pbar.close()
        return results

    def __call__(self, test_cases, task, load_data, solve_source, config_path, src_dir,
                          timeout=60, instance_workers=4, case_workers=4):
        return self.process_all_cases(test_cases, task, load_data, solve_source, config_path, src_dir,
                                      timeout=timeout, instance_workers=instance_workers, case_workers=case_workers)


def filter_dev(results, dev):
    if dev is None:
        return results
    dev_results = {}
    for case, (scores, error_message) in results.items():
        if case not in dev:
            continue
        dev_list = dev[case]
        if len(dev_list) == 0:
            dev_list = [0]
        select_scores = []
        for idx, score in enumerate(scores):
            if idx in dev_list:
                select_scores.append(score)
        if len(select_scores) > 0:
            dev_results[case] = (select_scores, error_message)
    return dev_results


def filter_test(results, dev):
    if dev is None:
        return results
    test_results = {}
    for case, (scores, error_message) in results.items():
        if case not in dev:
            test_results[case] = (scores, error_message)
            continue
        dev_list = dev[case]
        if len(dev_list) == 0:
            dev_list = [0]
        select_scores = []
        for idx, score in enumerate(scores):
            if idx not in dev_list:
                select_scores.append(score)
        if len(select_scores) > 0:
            test_results[case] = (select_scores, error_message)
    return test_results


def average_score(results, test_cases):
    return sum(
        (sum(x if not isinstance(x, str) else 0 for x in scores) / len(scores)
         if not error_message else 0)
        for scores, error_message in (results.get(case, (None, "No result")) for case in results.keys())
    ) / len(results)


def geo_men(results, test_cases):
    per_case_gms = []
    for case in results.keys():
        scores, error_message = results.get(case, (None, "No result"))
        if error_message:
            per_case_gms.append(0.0)
        else:
            # map non-str entries to themselves, str entries to 0
            vals = [x if not isinstance(x, str) else 0 for x in scores]
            k = len(vals)
            if k == 0:
                gm = 0.0
            else:
                prod = math.prod(vals)
                gm = prod**(1.0 / k)
            per_case_gms.append(gm)

    n = len(per_case_gms)
    if n == 0:
        return 0.0
    # overall geometric mean = (∏ per_case_gm)^(1/n)
    total_prod = math.prod(per_case_gms)
    return total_prod**(1.0 / n)

def compare_results(results, reference_results, test_cases):
    imp = dec = tie = 0
    for case in test_cases:
        new, new_err = results.get(case, (None, "No result"))
        ref, ref_err = reference_results.get(case, (None, "No result"))
        new_avg = sum(x if not isinstance(x, str) else 0 for x in new) / len(new) if not new_err else 0
        ref_avg = sum(x if not isinstance(x, str) else 0 for x in ref) / len(ref) if not ref_err else 0
        imp, dec, tie = (imp + 1, dec, tie) if new_avg > ref_avg else (imp, dec + 1, tie) if new_avg < ref_avg else (
        imp, dec, tie + 1)
    return imp, dec, tie


def extract_code_blocks(response):
    pattern_backticks = r"```python\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks

