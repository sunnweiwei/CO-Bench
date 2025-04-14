from evaluation.utils import FileLock, process_all_cases, design_optimal, eval_all, filter_dev, filter_test
import os
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



class Evaluator:
    def __init__(self, data, timeout=10, cpu_num=None, feedback_length=64):
        self.data = data
        self.timeout = timeout
        data_size = {case: [1] * len(
            self.data.load_data(f"{self.data.src_dir}/{self.data.task}/{case}")) for case in self.data.test_cases}
        cpu_num = os.cpu_count() or cpu_num
        self.case_workers, self.instance_workers = design_optimal(data_size, cpu_num)
        self.feedback_length = feedback_length

    def get_feedback(self, results, avg_score):
        prev_score = []
        for case in results.keys():
            scores, error_message = results.get(case, (None, "No result"))
            if error_message:
                prev_score.append(f"{case} -> Caught Error: {error_message}")
            else:
                # _scores = sorted(scores, key=lambda x: -1 if isinstance(x, str) else x)
                _scores = scores
                _scores = [x if isinstance(x, str) else f"{float(x):.3f}" for x in _scores][:self.feedback_length]
                prev_score.append(f"{case} -> Scores: {_scores}")
        # prev_score = sorted(prev_score, key=lambda x: -1 if isinstance(x[0], str) else 1)
        prev_score = '\n'.join(prev_score[:self.feedback_length])
        prev_score += f'\nAvg Score {avg_score}'
        return prev_score


    def evaluate(self, code):
        with FileLock():
            results = process_all_cases(
                self.data.test_cases, self.data.task, self.data.load_data, code,
                self.data.config_path, self.data.src_dir,
                timeout=self.timeout, instance_workers=self.instance_workers, case_workers=self.case_workers)
        results = self.data.norm_score(results)
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
            results=results
        )
