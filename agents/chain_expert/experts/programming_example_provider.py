from agents.chain_expert.experts.base_expert import BaseExpert


class ProgrammingExampleProvider(BaseExpert):

    ROLE_DESCRIPTION = 'You are a Python programmer in the field of operations research and optimization. Your proficiency in utilizing third-party libraries such as Gurobi is essential. In addition to your expertise in Gurobi, it would be great if you could also provide some background in related libraries or tools, like NumPy, SciPy, or PuLP.'
    
    FORWARD_TASK = '''You are given a specific problem. You aim to develop an efficient Python program that addresses the given problem.
Now the origin problem is as follow:
{problem_description}
Let's analyse the problem step by step, and then give your Python code.
Here is a starter code:
{code_example}
And the comments from other experts are as follow:
{comments_text}

Give your Python code directly.'''
    
    BACKWARD_TASK = '''When you are solving a problem, you get a feedback from the external environment. You need to judge whether this is a problem caused by you or by other experts (other experts have given some results before you). If it is your problem, you need to give Come up with solutions and refined code.

The original problem is as follow:
{problem_description}

The feedback is as follow:
{feedback}

The modeling you give previously is as follow:
{previous_modeling}

The output format is a JSON structure followed by refined code:
{{
    "is_caused_by_you": false,
    "reason": "leave empty string if the problem is not caused by you",
    "refined_result": "Your refined result"
}}
'''

    def __init__(self, model):
        super().__init__(
            name='Programming Expert',
            description='Skilled in programming and coding, capable of implementing the optimization solution in a programming language.',
            model=model   
        )
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK

    def forward(self, problem, comment_pool):
        self.problem = problem
        comments_text = comment_pool.get_current_comment_text()
        output = self.predict(self.forward_prompt_template,
            problem_description=problem['description'], 
            comments_text=comments_text
        )
        self.previous_answer = output
        return output

    def backward(self, feedback_pool):
        if not hasattr(self, 'problem'):
            raise NotImplementedError('Please call forward first!')
        output = self.predict(self.backward_prompt_template,
            problem_description=self.problem['description'], 
            previous_answer=self.previous_answer,
            feedback=feedback_pool.get_current_comment_text())
        return output
