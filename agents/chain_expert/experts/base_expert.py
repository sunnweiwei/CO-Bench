from agents.chain_expert.utils import call_llm


class BaseExpert(object):

    def __init__(self, name, description, model):
        self.name = name
        self.description = description
        self.model = model
        self.temperature = 0
        self.max_tokens = 4096
        
        self.forward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.FORWARD_TASK
        
        if hasattr(self, 'BACKWARD_TASK'):
            self.backward_prompt_template = self.ROLE_DESCRIPTION + '\n' + self.BACKWARD_TASK

    def forward(self):
        pass

    def backward(self):
        pass
    
    def format_prompt(self, template, **kwargs):
        """Format a prompt template with the given kwargs."""
        formatted_prompt = template
        for key, value in kwargs.items():
            formatted_prompt = formatted_prompt.replace('{' + key + '}', str(value))
        return formatted_prompt
    
    def predict(self, prompt_template, **kwargs):
        """Make a prediction using the LLM with the given prompt template and kwargs."""
        prompt = self.format_prompt(prompt_template, **kwargs)
        return call_llm(self.model, prompt, self.temperature, self.max_tokens)

    def __str__(self):
        return f'{self.name}: {self.description}'
