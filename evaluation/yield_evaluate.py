from evaluation.evaluate import Evaluator, Feedback
from evaluation.utils import FileLock, ParallelRun, average_score, filter_dev, filter_test, import_func
import time
import os
import multiprocessing as mp
import psutil
import traceback
import sys
import re
import signal


def format_concise_error(exc_type, exc_value, exc_traceback):
    """Format a concise error message with just the essential information."""
    # Get the full traceback as a string
    tb_lines = traceback.format_exception(exc_type, exc_value, exc_traceback)

    # Look for the solve function line in the traceback
    for line in tb_lines:
        if ", in solve" in line:
            # Extract file, line number and function name
            match = re.search(r'line (\d+), in (\w+)', line)
            if match:
                line_num, func_name = match.groups()
                return f"line {line_num}, in {func_name} {exc_type.__name__}: {str(exc_value)}"

    # If we couldn't find a specific solve line, return a simplified message
    return f"{exc_type.__name__}: {str(exc_value)}"


def evaluate_yielding_instance_in_subprocess(instance, solve_source, config_path, queue):
    """
    Run evaluation inside a process and continuously send yielded solutions to the parent process.
    """
    try:
        # Set process group ID to make it easier to kill all children later
        if hasattr(os, 'setpgrp'):  # Unix/Linux/Mac
            os.setpgrp()

        # Re-import eval_func from the config file
        _, eval_func = import_func(config_path, 'load_data', 'eval_func')

        # Compile the solve function from its source code
        local_namespace = {}
        exec(solve_source, local_namespace)
        if "solve" not in local_namespace:
            raise ValueError("The source code does not define a 'solve' function.")
        solve_func = local_namespace["solve"]

        # Call solve function and collect solutions
        # print(len(instance['terminals']))
        try:
            for solution in solve_func(**instance):
                queue.put(("solution", solution))
        except Exception as e:
            # Get concise error message
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_message = format_concise_error(exc_type, exc_value, exc_traceback)
            queue.put(("error", f"Exception during solving: {error_message}"))
    except Exception as e:
        # Get concise error message for setup errors
        exc_type, exc_value, exc_traceback = sys.exc_info()
        error_message = format_concise_error(exc_type, exc_value, exc_traceback)
        queue.put(("error", f"Exception in setup: {error_message}"))


def run_yielding_instance_with_timeout(instance, solve_source, config_path, timeout):
    """
    Run instance with timeout, collecting yielded solutions from the subprocess.
    After the subprocess finishes or times out, evaluate the last solution.
    """
    queue = mp.Queue()
    p = mp.Process(target=evaluate_yielding_instance_in_subprocess,
                   args=(instance, solve_source, config_path, queue))
    p.start()

    last_solution = None
    error = None

    # Wait for the subprocess to finish or time out
    end_time = time.time() + timeout
    while time.time() < end_time and p.is_alive():
        # Check for new data from the subprocess
        try:
            while not queue.empty() and time.time() < end_time:
                message_type, data = queue.get_nowait()
                if message_type == "solution":
                    last_solution = data
                elif message_type == "error":
                    error = data
        except Exception:
            pass
        # Sleep to prevent CPU spinning
        time.sleep(0.05)
    # print('Done')
    # If the process is still alive, terminate it
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
    # print('Killed')
    # if p.is_alive():
    #     # print(f"WARNING: Process {p.pid} could not be terminated!")
    #     # Last resort: use system kill command on Unix
    #     if hasattr(os, 'system'):
    #         os.system(f"kill -9 {p.pid} 2>/dev/null || true")
    # print('Final Killed')

    # Create a thread to do the queue fetching with the original code
    import threading

    def fetch_from_queue():
        """Fetch remaining data from the queue using the original method."""
        nonlocal last_solution, error
        try:
            while not queue.empty():
                message_type, data = queue.get_nowait()
                if message_type == "solution":
                    last_solution = data
                elif message_type == "error":
                    error = data
        except Exception:
            pass

    # Run the queue fetching in a separate thread with timeout
    fetch_thread = threading.Thread(target=fetch_from_queue)
    fetch_thread.daemon = True
    fetch_thread.start()
    fetch_thread.join(timeout=2.0)

    # If there was an error and no solution, return the error
    if error and not last_solution:
        return error

    # If we have a last solution, evaluate it
    if last_solution is not None:
        # Re-import eval_func from the config file
        _, eval_func = import_func(config_path, 'load_data', 'eval_func')

        # Convert to string keys for consistency
        last_solution = {str(k): v for k, v in last_solution.items()}

        # Evaluate the last solution
        try:
            score = eval_func(**instance, **last_solution)
            return score
        except Exception as e:
            # Get full traceback for evaluation errors
            exc_type, exc_value, exc_traceback = sys.exc_info()
            error_message = f"{exc_type.__name__}: {str(exc_value)}"
            return f"{error_message}"
    else:
        # No solution was yielded
        return f"Timeout ({timeout}s) with no solution yielded"


class YieldingParallelRun(ParallelRun):
    def __init__(self, *args, **kwargs):
        super().__init__(None, *args, **kwargs)

    def evaluate_instance_in_subprocess(self, instance, solve_source, config_path, queue):
        """
        Override the subprocess evaluation to handle yielding solve functions.
        """
        evaluate_yielding_instance_in_subprocess(instance, solve_source, config_path, queue)

    def run_instance_with_timeout(self, instance, solve_source, config_path, timeout):
        """
        Override the timeout handling to collect yielded solutions.
        """
        return run_yielding_instance_with_timeout(instance, solve_source, config_path, timeout)


class YieldingEvaluator(Evaluator):
    """
    An evaluator that handles solve functions that yield multiple solutions.
    It collects the last solution yielded before timeout and evaluates it after the timeout.
    """

    def __init__(self, data, timeout=10, cpu_num=None, feedback_length=64):
        super().__init__(data, timeout, cpu_num, feedback_length)

    def get_feedback(self, results, avg_score):
        grouped = {}
        for case, (scores, err) in results.items():
            key = case.split("/", 1)[0] if "/" in case else case
            bucket = grouped.setdefault(key, {"scores": [], "errors": []})

            if err:
                bucket["errors"].append(f"{case.split('/')[-1]}: {err}")
            else:
                bucket["scores"].extend(
                    x if isinstance(x, str) else f"{float(x):.3f}" for x in scores
                )

        lines = []
        for key, data in grouped.items():
            if data["scores"]:
                lines.append(f"{key} -> Scores: {data['scores'][:self.feedback_length]}")
            if data["errors"]:
                lines.append(f"{key} -> Errors: {data['errors'][:self.feedback_length]}")

        summary = "\n".join(lines[: self.feedback_length])
        summary += f"\nAvg Score {avg_score}"
        return summary

    def evaluate(self, code):
        runtime = YieldingParallelRun()
        results = runtime(
            self.data.test_cases, self.data.task, self.data.load_data, code,
            self.data.config_path, self.data.src_dir,
            timeout=self.timeout, instance_workers=self.instance_workers, case_workers=self.case_workers)
        results = self.data.norm_score(results)
        score = average_score(results, self.data.test_cases)
        dev_score = average_score(filter_dev(results, self.data.get_dev()), self.data.test_cases)
        test_score = average_score(filter_test(results, self.data.get_dev()), self.data.test_cases)

        feedback = self.get_feedback(results, dev_score)
        dev_feedback = self.get_feedback(filter_dev(results, self.data.get_dev()), dev_score)
        test_feedback = self.get_feedback(filter_test(results, self.data.get_dev()), test_score)
        return Feedback(
            score=score,
            dev_score=dev_score,
            test_score=test_score,
            feedback=feedback,
            dev_feedback=dev_feedback,
            test_feedback=test_feedback,
            results=results
        )