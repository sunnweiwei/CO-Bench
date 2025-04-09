import re
import textwrap


def call_llm(question: str, model='openai/gpt-4o', reasoning_effort=None) -> str:
    """
    Call a language model with a question and return the response.
    
    Args:
        question (str): The question to ask the language model
        model (str, optional): The model to use. Defaults to 'openai/gpt-4o'.
        reasoning_effort (any, optional): Optional parameter for reasoning effort. Defaults to None.
        
    Returns:
        str: The model's response
    """
    import litellm
    from litellm import completion
    litellm.drop_params = True
    messages = [{"content": question, "role": "user"}]
    response = completion(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.choices[0].message.content


def extract_code_blocks(response):
    """
    Extract code blocks from a response string.
    
    Args:
        response (str): The response string containing code blocks
        
    Returns:
        list: A list of extracted code blocks
    """
    pattern_backticks = r"```python\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks
