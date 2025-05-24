from evaluation.evaluate import Evaluator, Feedback
from evaluation.utils import import_func, average_score, filter_dev, filter_test
import threading
import time
import os


class SimpleYieldingEvaluator(Evaluator):
    """
    A simplified evaluator that handles solve functions that yield multiple solutions.
    It collects the last solution yielded before timeout and evaluates it.
    """

    def evaluate_yielding_instance(self, instance, solve, eval_func, timeout=10):
        """
        Run solve (as a generator) and collect yielded solutions until timeout.
        Then evaluate the last solution.
        """
        last_solution = [None]  # Use a list to make it mutable from the thread
        error = [None]
        try:
            for solution in solve(**instance):
                print(solution)
                last_solution[0] = solution
        except Exception as e:
            print(e)
            pass

        # If there was an error and no solution, raise the error
        if error[0] and last_solution[0] is None:
            raise RuntimeError(error[0])

        # If no solution was yielded, consider it a timeout
        if last_solution[0] is None:
            return f"Timeout ({timeout}s) with no solution yielded"

        # Evaluate the last solution
        solution = {str(k): v for k, v in last_solution[0].items()}
        score = eval_func(**instance, **solution)
        # print(score)
        return score

    def evaluate(self, code):
        # Compile the solve function
        namespace = {}
        print(code)
        exec(code, namespace)
        if "solve" not in namespace:
            raise ValueError("The source code does not define a 'solve' function.")
        solve = namespace["solve"]

        # Re-import eval_func from the config file
        _, eval_func = import_func(self.data.config_path, 'load_data', 'eval_func')

        all_results = {}

        # Process each test case
        from tqdm import tqdm
        for case in tqdm(self.data.test_cases):
            file_path = os.path.join(self.data.src_dir, self.data.task, case)
            instances = self.data.load_data(file_path)

            case_results = []
            error_message = None

            # Process each instance
            for instance in instances:
                result = self.evaluate_yielding_instance(
                    instance, solve, eval_func, self.timeout)
                case_results.append(result)

            # print(result)
            all_results[case] = (case_results, error_message)

        # Apply normalization
        all_results = self.data.norm_score(all_results)

        # Calculate scores
        score = average_score(all_results, self.data.test_cases)
        dev_score = average_score(filter_dev(all_results, self.data.get_dev()), self.data.test_cases)
        test_score = average_score(filter_test(all_results, self.data.get_dev()), self.data.test_cases)

        # Generate feedback
        feedback = self.get_feedback(all_results, score)
        dev_feedback = self.get_feedback(filter_dev(all_results, self.data.get_dev()), dev_score)
        test_feedback = self.get_feedback(filter_test(all_results, self.data.get_dev()), test_score)

        return Feedback(
            score=score,
            dev_score=dev_score,
            test_score=test_score,
            feedback=feedback,
            dev_feedback=dev_feedback,
            test_feedback=test_feedback,
            results=all_results
        )