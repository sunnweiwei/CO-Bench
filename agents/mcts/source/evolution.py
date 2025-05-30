import re
import time
from agents.mcts.source.interface_LLM import InterfaceAPI as InterfaceLLM
import re

input = lambda: ...

def extract_code_blocks(response):
    pattern_backticks = r"```python\s*(.*?)\s*```"
    pattern_dashes = r"^-{3,}\s*\n(.*?)\n-{3,}"
    blocks = re.findall(pattern_backticks, response, re.DOTALL)
    blocks.extend(re.findall(pattern_dashes, response, re.DOTALL | re.MULTILINE))
    return blocks

class Evolution():

    def __init__(self, api_endpoint, api_key, model_LLM, debug_mode, prompts, **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        self._use_local_llm = kwargs.get('use_local_llm')
        self._url = kwargs.get('url')
        # -----------------------------------------------------------

        # set prompt interface
        # getprompts = GetPrompts()
        self.prompt_task = prompts.get_task()
        self.prompt_func_name = prompts.get_func_name()
        self.prompt_func_inputs = prompts.get_func_inputs()
        self.prompt_func_outputs = prompts.get_func_outputs()
        self.prompt_inout_inf = prompts.get_inout_inf()
        self.prompt_other_inf = prompts.get_other_inf()
        if len(self.prompt_func_inputs) > 1:
            self.joined_inputs = ", ".join("'" + s + "'" for s in self.prompt_func_inputs)
        else:
            self.joined_inputs = "'" + self.prompt_func_inputs[0] + "'"

        if len(self.prompt_func_outputs) > 1:
            self.joined_outputs = ", ".join("'" + s + "'" for s in self.prompt_func_outputs)
        else:
            self.joined_outputs = "'" + self.prompt_func_outputs[0] + "'"

        # set LLMs
        self.api_endpoint = api_endpoint
        self.api_key = api_key
        self.model_LLM = model_LLM
        self.debug_mode = debug_mode  # close prompt checking

        # -------------------- RZ: use local LLM --------------------
        if self._use_local_llm:
            self.interface_llm = LocalLLM(self._url)
        else:
            self.interface_llm = InterfaceLLM(self.api_endpoint, self.api_key, self.model_LLM, self.debug_mode)

    def get_prompt_post(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the a Code implementing a heuristic algorithm with function name " + self.prompt_func_name + " to solve the above mentioned problem.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nNow you should describe the Design Idea of the algorithm using less than 5 sentences.\n"
        prompt_content += "Hint: You should highlight every meaningful designs in the provided code and describe their ideas. You can analyse the code to see which variables are given higher values and which variables are given lower values, the choice of parameters or the total structure of the code."
        return prompt_content

    def get_prompt_refine(self, code, algorithm):

        prompt_content = self.prompt_task + "\n" + "Following is the Design Idea of a heuristic algorithm for the problem and the code with function name '" + self.prompt_func_name + "' for implementing the heuristic algorithm.\n"
        prompt_content += self.prompt_inout_inf + " " + self.prompt_other_inf
        prompt_content += "\nDesign Idea:\n" + algorithm
        prompt_content += "\n\nCode:\n" + code
        prompt_content += "\n\nThe content of the Design Idea idea cannot fully represent what the algorithm has done informative. So, now you should re-describe the algorithm using less than 3 sentences.\n"
        prompt_content += "Hint: You should reference the given Design Idea and highlight the most critical design ideas of the code. You can analyse the code to describe which variables are given higher priorities and which variables are given lower priorities, the parameters and the structure of the code."
        return prompt_content

    def get_prompt_i1(self):

        prompt_content = self.prompt_task + "\n" + "First, describe the design idea and main steps of your algorithm in one sentence. " + "The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes as follows: \n\n" \
                         + prompt_indiv + \
                         "Please create a new algorithm that has a totally different form from the given algorithms. Try generating codes with different structures, flows or algorithms. The new algorithm should have a relatively low objective value. \n" \
                         "First, describe the design idea and main steps of your algorithm in one sentence. The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_e2(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            # print(indivs[i]['algorithm'] + f"Objective value: {indivs[i]['objective']}")
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please create a new algorithm that has a similar form to the No.{len(indivs)} algorithm and is inspired by the No.{1} algorithm. The new algorithm should have a objective value lower than both algorithms.\n" \
                         f"Firstly, list the common ideas in the No.{1} algorithm that may give good performances. Secondly, based on the common idea, describe the design idea based on the No.{len(indivs)} algorithm and main steps of your algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m1(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please create a new algorithm that has a different form but can be a modified version of the provided algorithm. Attempt to introduce more novel mechanisms and new equations or programme segments.\n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_m2(self, indiv1):
        prompt_content = self.prompt_task + "\n" \
                                            "I have one algorithm with its code as follows. \n\n\
Algorithm's description: " + indiv1['algorithm'] + "\n\
Code:\n\
" + indiv1['code'] + "\n\
Please identify the main algorithm parameters and help me in creating a new algorithm that has different parameter settings to equations compared to the provided algorithm. \n" \
                     "First, describe the design idea based on the provided algorithm and main steps of the new algorithm in one sentence. \
The description must be inside a brace outside the code implementation. Next, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def get_prompt_s1(self, indivs):
        prompt_indiv = ""
        for i in range(len(indivs)):
            prompt_indiv = prompt_indiv + "No." + str(
                i + 1) + " algorithm's description, its corresponding code and its objective value are: \n" + \
                           indivs[i]['algorithm'] + "\n" + indivs[i][
                               'code'] + "\n" + f"Objective value: {indivs[i]['objective']}" + "\n\n"

        prompt_content = self.prompt_task + "\n" \
                                            "I have " + str(
            len(indivs)) + " existing algorithms with their codes and objective values as follows: \n\n" \
                         + prompt_indiv + \
                         f"Please help me create a new algorithm that is inspired by all the above algorithms with its objective value lower than any of them.\n" \
                         "Firstly, list some ideas in the provided algorithms that are clearly helpful to a better algorithm. Secondly, based on the listed ideas, describe the design idea and main steps of your new algorithm in one sentence. \
The description must be inside a brace. Thirdly, implement it in Python as a function named \
'" + self.prompt_func_name + "'.\nThis function should accept " + str(
            len(self.prompt_func_inputs)) + " input(s): " \
                         + self.joined_inputs + ". The function should return " + str(
            len(self.prompt_func_outputs)) + " output(s): " \
                         + self.joined_outputs + ". " + self.prompt_inout_inf + " " \
                         + self.prompt_other_inf + "\n" + "Do not give additional explanations."
        return prompt_content

    def _get_thought(self, prompt_content):

        response = self.interface_llm.get_response(prompt_content, 0)

        # algorithm = response.split(':')[-1]
        return response

    def _get_alg(self, prompt_content):
        import textwrap

        response = self.interface_llm.get_response(prompt_content)

        code_blocks = extract_code_blocks(response)
        code = textwrap.dedent(code_blocks[0])
        algorithm = response.split('\n')[0]
        return [code, algorithm]

        response = self.interface_llm.get_response(prompt_content)

        algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
        if len(algorithm) == 0:
            if 'python' in response:
                algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
            elif 'import' in response:
                algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
            else:
                algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

        code = re.findall(r"import.*return", response, re.DOTALL)
        if len(code) == 0:
            code = re.findall(r"def.*return", response, re.DOTALL)

        n_retry = 1
        while (len(algorithm) == 0 or len(code) == 0):
            if self.debug_mode:
                print("Error: algorithm or code not identified, wait 1 seconds and retrying ... ")

            response = self.interface_llm.get_response(prompt_content)

            algorithm = re.search(r"\{(.*?)\}", response, re.DOTALL).group(1)
            if len(algorithm) == 0:
                if 'python' in response:
                    algorithm = re.findall(r'^.*?(?=python)', response, re.DOTALL)
                elif 'import' in response:
                    algorithm = re.findall(r'^.*?(?=import)', response, re.DOTALL)
                else:
                    algorithm = re.findall(r'^.*?(?=def)', response, re.DOTALL)

            code = re.findall(r"import.*return", response, re.DOTALL)
            if len(code) == 0:
                code = re.findall(r"def.*return", response, re.DOTALL)

            if n_retry > 3:
                break
            n_retry += 1

        code = code[0]
        code_all = code + " " + ", ".join(s for s in self.prompt_func_outputs)

        return [code_all, algorithm]

    def post_thought(self, code, algorithm):

        prompt_content = self.get_prompt_refine(code, algorithm)

        post_thought = self._get_thought(prompt_content)

        return post_thought

    def i1(self):

        prompt_content = self.get_prompt_i1()

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ i1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e1(self, parents):

        prompt_content = self.get_prompt_e1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def e2(self, parents):

        prompt_content = self.get_prompt_e2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ e2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m1(self, parents):

        prompt_content = self.get_prompt_m1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def m2(self, parents):

        prompt_content = self.get_prompt_m2(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ m2 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]

    def s1(self, parents):

        prompt_content = self.get_prompt_s1(parents)

        if self.debug_mode:
            print("\n >>> check prompt for creating algorithm using [ s1 ] : \n", prompt_content)
            print(">>> Press 'Enter' to continue")
            input()

        [code_all, algorithm] = self._get_alg(prompt_content)

        if self.debug_mode:
            print("\n >>> check designed algorithm: \n", algorithm)
            print("\n >>> check designed code: \n", code_all)
            print(">>> Press 'Enter' to continue")
            input()

        return [code_all, algorithm]
