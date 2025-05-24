import re
import textwrap
import importlib.util
import os


def call_llm(question: str, model='openai/gpt-4o', reasoning_effort=None) -> str:
    import litellm
    from litellm import completion
    litellm.drop_params = True
    messages = [{"content": question, "role": "user"}]
    response = completion(model=model, messages=messages, reasoning_effort=reasoning_effort)
    return response.choices[0].message.content


def extract_code_blocks(response):
    pattern_backticks = r"```python\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks
