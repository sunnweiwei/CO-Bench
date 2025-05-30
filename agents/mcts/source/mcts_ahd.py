import copy

import numpy as np
import json
import random
import time

from agents.mcts.source.mcts import MCTS, MCTSNode
from agents.mcts.source.evolution_interface import InterfaceEC


# main class for eoh
class MCTS_AHD:
    def __init__(self, paras, problem, select, manage, **kwargs):

        self.prob = problem
        self.select = select
        self.manage = manage
        # LLM settings
        self.use_local_llm = paras.llm_use_local
        self.url = paras.llm_local_url
        self.api_endpoint = paras.llm_api_endpoint  # currently only API2D + GPT
        self.api_key = paras.llm_api_key
        self.llm_model = paras.llm_model

        # ------------------ RZ: use local LLM ------------------
        self.use_local_llm = kwargs.get('use_local_llm', False)
        assert isinstance(self.use_local_llm, bool)
        if self.use_local_llm:
            assert 'url' in kwargs, 'The keyword "url" should be provided when use_local_llm is True.'
            assert isinstance(kwargs.get('url'), str)
            self.url = kwargs.get('url')
        # -------------------------------------------------------

        # Experimental settings       
        self.init_size = paras.init_size  # popopulation size, i.e., the number of algorithms in population
        self.pop_size = paras.pop_size  # popopulation size, i.e., the number of algorithms in population
        self.fe_max = paras.ec_fe_max  # function evaluation times
        self.eval_times = 0  # number of populations

        self.operators = paras.ec_operators
        self.operator_weights = paras.ec_operator_weights
        paras.ec_m = 5
        self.m = paras.ec_m

        self.debug_mode = paras.exp_debug_mode  # if debug
        self.ndelay = 1  # default

        self.use_seed = paras.exp_use_seed
        self.seed_path = paras.exp_seed_path
        self.load_pop = paras.exp_use_continue
        self.load_pop_path = paras.exp_continue_path
        self.load_pop_id = paras.exp_continue_id

        self.output_path = paras.exp_output_path

        self.exp_n_proc = paras.exp_n_proc

        self.timeout = paras.eva_timeout

        self.use_numba = paras.eva_numba_decorator

        print("- MCTS-AHD parameters loaded -")

        # Set a random seed
        random.seed(2024)

    # add new individual to population
    def add2pop(self, population, offspring):
        for ind in population:
            if ind['algorithm'] == offspring['algorithm']:
                if (self.debug_mode):
                    print("duplicated result, retrying ... ")
        population.append(offspring)

    def expand(self, mcts, cur_node, nodes_set, option):
        if option == 's1':
            path_set = []
            now = copy.deepcopy(cur_node)
            while now.code != "Root":
                path_set.append(now.raw_info)
                now = copy.deepcopy(now.parent)
            path_set = self.manage.population_management_s1(path_set, len(path_set))
            if len(path_set) == 1:
                return nodes_set
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, path_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        elif option == 'e1':
            e1_set = [copy.deepcopy(children.subtree[random.choices(range(len(children.subtree)), k=1)[0]].raw_info) for
                      children in mcts.root.children]
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, e1_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        else:
            self.eval_times, offsprings = self.interface_ec.evolve_algorithm(self.eval_times, nodes_set,
                                                                             cur_node.raw_info,
                                                                             cur_node.children_info, option)
        if offsprings == None:
            print(f"Timeout emerge, no expanding with action {option}.")
            return nodes_set

        if option != 'e1':
            print(
                f"Action: {option}, Father Obj: {cur_node.raw_info['objective']}, Now Obj: {offsprings['objective']}, Depth: {cur_node.depth + 1}")
        else:
            if self.interface_ec.check_duplicate_obj(mcts.root.children_info, offsprings['objective']):
                print(f"Duplicated e1, no action, Father is Root, Abandon Obj: {offsprings['objective']}")
                return nodes_set
            else:
                print(f"Action: {option}, Father is Root, Now Obj: {offsprings['objective']}")
        if offsprings['objective'] != float('inf'):
            self.add2pop(nodes_set, offsprings)  # Check duplication, and add the new offspring
            size_act = min(len(nodes_set), self.pop_size)
            nodes_set = self.manage.population_management(nodes_set, size_act)
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'],
                               parent=cur_node, depth=cur_node.depth + 1,
                               visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
            if option == 'e1':
                nownode.subtree.append(nownode)
            cur_node.add_child(nownode)
            cur_node.children_info.append(offsprings)
            mcts.backpropagate(nownode)
        return nodes_set

    # run eoh
    def run(self):
        print("- Initialization Start -")

        interface_prob = self.prob

        # interface for ec operators
        self.interface_ec = InterfaceEC(self.m, self.api_endpoint, self.api_key, self.llm_model,
                                        self.debug_mode, interface_prob, use_local_llm=self.use_local_llm, url=self.url,
                                        select=self.select, n_p=self.exp_n_proc,
                                        timeout=self.timeout, use_numba=self.use_numba
                                        )

        brothers = []
        mcts = MCTS('Root')
        # main loop
        n_op = len(self.operators)
        self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(self.eval_times, brothers, "i1")
        brothers.append(offsprings)
        nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'], parent=mcts.root,
                           depth=1, visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
        mcts.root.add_child(nownode)
        mcts.root.children_info.append(offsprings)
        mcts.backpropagate(nownode)
        nownode.subtree.append(nownode)
        for i in range(1, self.init_size):
            self.eval_times, brothers, offsprings = self.interface_ec.get_algorithm(self.eval_times, brothers, "e1")
            brothers.append(offsprings)
            nownode = MCTSNode(offsprings['algorithm'], offsprings['code'], offsprings['objective'], parent=mcts.root,
                               depth=1, visit=1, Q=-1 * offsprings['objective'], raw_info=offsprings)
            mcts.root.add_child(nownode)
            mcts.root.children_info.append(offsprings)
            mcts.backpropagate(nownode)
            nownode.subtree.append(nownode)
        nodes_set = brothers
        size_act = min(len(nodes_set), self.pop_size)
        nodes_set = self.manage.population_management(nodes_set, size_act)
        print("- Initialization Finished - Evolution Start -")
        while self.eval_times < self.fe_max:
            print(f"Current performances of MCTS nodes: {mcts.rank_list}")
            # print([len(node.subtree) for node in mcts.root.children])
            cur_node = mcts.root
            while len(cur_node.children) > 0 and cur_node.depth < mcts.max_depth:
                uct_scores = [mcts.uct(node, max(1 - self.eval_times / self.fe_max, 0)) for node in cur_node.children]
                selected_pair_idx = uct_scores.index(max(uct_scores))
                if int((cur_node.visits) ** mcts.alpha) > len(cur_node.children):
                    if cur_node == mcts.root:
                        op = 'e1'
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                    else:
                        # i = random.randint(1, n_op - 1)
                        i = 1
                        op = self.operators[i]
                        nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                cur_node = cur_node.children[selected_pair_idx]
            for i in range(n_op):
                op = self.operators[i]
                print(f"Iter: {self.eval_times}/{self.fe_max} OP: {op}", end="|")
                op_w = self.operator_weights[i]
                for j in range(op_w):
                    nodes_set = self.expand(mcts, cur_node, nodes_set, op)
                assert len(cur_node.children) == len(cur_node.children_info)
            # Save population to a file
            filename = self.output_path + "population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set, f, indent=5)

            # Save the best one to a file
            filename = self.output_path + "best_population_generation_" + str(self.eval_times) + ".json"
            with open(filename, 'w') as f:
                json.dump(nodes_set[0], f, indent=5)

        return nodes_set[0]["code"], filename
