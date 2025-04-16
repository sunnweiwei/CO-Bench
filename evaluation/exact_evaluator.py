from evaluation.utils import FileLock, ParallelRun, design_optimal, eval_all, filter_dev, filter_test
import time
from evaluation.evaluate import Evaluator
from dataclasses import dataclass

@dataclass
class Feedback:
    score: float
    dev_score: float
    test_score: float
    feedback: str
    dev_feedback: str
    test_feedback: str
    results: dict


def separate_time(results):
    scores = {}
    times = {}
    for case, (scores, error_message) in results.items():
        score_item = []
        time_item = []
        for idx, score in enumerate(scores):
            if isinstance(score, list):
                score_item.append(score[0])
                time_item.append(score[1])
            else:
                score_item.append(score)
                time_item.append(score)
        scores[case] = (score_item, error_message)
        times[case] = (time_item, error_message)
    return scores, times


def optimal_filter(results):
    normed = {}
    for case, (scores, error_message) in results.items():
        normed_scores = []
        for idx, score in enumerate(scores):
            if isinstance(score, (int, float)):
                normed_scores.append(float(score >= 1.0))
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)
    return normed


def filter_time(score_results, time_results):
    normed = {}
    for case in score_results:
        scores, error_message = score_results[case]
        times = time_results[case]
        normed_scores = []
        for score, t in zip(scores, times):
            if isinstance(score, (int, float)):
                if score >= 0.0:
                    normed_scores.append(t)
                else:
                    normed_scores.append(0.0)
            else:
                normed_scores.append(score)
        normed[case] = (normed_scores, error_message)
    return normed


def evaluate_instance(instance, solve, eval_func):
    """Run solve and eval_func on the instance and return the score."""
    start_time = time.time()
    solution = solve(**instance)
    cost = time.time() - start_time
    solution = {str(k): v for k, v in solution.items()}
    score = eval_func(**instance, **solution)
    return [score, cost]


class ExactEvaluator(Evaluator):
    def evaluate(self, code):
        runtime = ParallelRun(evaluate_instance)
        with FileLock():
            results = runtime(
                self.data.test_cases, self.data.task, self.data.load_data, code,
                self.data.config_path, self.data.src_dir,
                timeout=self.timeout, instance_workers=self.instance_workers, case_workers=self.case_workers)
        score_results, time_results = separate_time(results)
        score_results = self.data.norm_score(score_results)
        score_results = optimal_filter(score_results)
        time_results = self.data.norm_time(results)

        results = filter_time(score_results, time_results)

        results = {k:v for k,v in results.items()}
        score = eval_all(results, self.data.test_cases)
        dev_score = eval_all(filter_dev(results, self.data.get_dev()), self.data.test_cases)
        test_score = eval_all(filter_test(results, self.data.get_dev()), self.data.test_cases)

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
            results=results,
        )
