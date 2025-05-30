import copy
import random

import numpy as np
import time

from agents.mcts.source.evolution import Evolution
import warnings
from joblib import Parallel, delayed
import re
import concurrent.futures


class InterfaceEC():
    def __init__(self, m, api_endpoint, api_key, llm_model, debug_mode, interface_prob, select, n_p, timeout, use_numba,
                 **kwargs):
        assert 'use_local_llm' in kwargs
        assert 'url' in kwargs
        # -----------------------------------------------------------

        # LLM settings
        self.interface_eval = interface_prob
        prompts = interface_prob.prompts
        self.evol = Evolution(api_endpoint, api_key, llm_model, debug_mode, prompts, **kwargs)
        self.m = m
        self.debug = debug_mode

        if not self.debug:
            warnings.filterwarnings("ignore")

        self.select = select
        self.n_p = n_p

        self.timeout = timeout
        self.use_numba = use_numba

    def code2file(self, code):
        with open("./ael_alg.py", "w") as file:
            # Write the code to the file
            file.write(code)
        return

    def add2pop(self, population, offspring):
        for ind in population:
            if ind['objective'] == offspring['objective']:
                if self.debug:
                    print("duplicated result, retrying ... ")
                return False
        population.append(offspring)
        return True

    def check_duplicate_obj(self, population, obj):
        for ind in population:
            if obj == ind['objective']:
                return True
        return False

    def check_duplicate(self, population, code):
        for ind in population:
            if code == ind['code']:
                return True
        return False

    def population_generation_seed(self, seeds):

        population = []

        fitness = self.interface_eval.batch_evaluate([seed['code'] for seed in seeds])

        for i in range(len(seeds)):
            try:
                seed_alg = {
                    'algorithm': seeds[i]['algorithm'],
                    'code': seeds[i]['code'],
                    'objective': None,
                    'other_inf': None
                }

                obj = np.array(fitness[i])
                seed_alg['objective'] = np.round(obj, 5)
                population.append(seed_alg)

            except Exception as e:
                print("Error in seed algorithm")
                exit()

        print("Initiliazation finished! Get " + str(len(seeds)) + " seed algorithms")

        return population

    def _get_alg(self, pop, operator, father=None):
        offspring = {
            'algorithm': None,
            'thought': None,
            'code': None,
            'objective': None,
            'other_inf': None
        }
        if operator == "i1":
            parents = None
            [offspring['code'], offspring['thought']] = self.evol.i1()
        elif operator == "e1":
            real_m = random.randint(2, self.m)
            real_m = min(real_m, len(pop))
            parents = self.select.parent_selection_e1(pop, real_m)
            [offspring['code'], offspring['thought']] = self.evol.e1(parents)
        elif operator == "e2":
            other = copy.deepcopy(pop)
            if father in pop:
                other.remove(father)
            real_m = 1
            # real_m = random.randint(2, self.m) - 1
            # real_m = min(real_m, len(other))
            parents = self.select.parent_selection(other, real_m)
            parents.append(father)
            [offspring['code'], offspring['thought']] = self.evol.e2(parents)
        elif operator == "m1":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m1(parents[0])
        elif operator == "m2":
            parents = [father]
            [offspring['code'], offspring['thought']] = self.evol.m2(parents[0])
        elif operator == "s1":
            parents = pop
            [offspring['code'], offspring['thought']] = self.evol.s1(pop)
        else:
            print(f"Evolution operator [{operator}] has not been implemented ! \n")

        offspring['algorithm'] = self.evol.post_thought(offspring['code'], offspring['thought'])
        return parents, offspring

    def get_offspring(self, pop, operator, father=None):
        while True:
            try:
                p, offspring = self._get_alg(pop, operator, father=father)
                code = offspring['code']
                n_retry = 1
                while self.check_duplicate(pop, offspring['code']):
                    n_retry += 1
                    if self.debug:
                        print("duplicated code, wait 1 second and retrying ... ")
                    p, offspring = self._get_alg(pop, operator, father=father)
                    code = offspring['code']
                    if n_retry > 1:
                        break
                break
            except Exception as e:
                print(e)
        return p, offspring

    def get_algorithm(self, eval_times, pop, operator):
        while True:
            eval_times += 1
            parents, offspring = self.get_offspring(pop, operator)
            objs = self.interface_eval.batch_evaluate([offspring['code']], 0)
            # if objs == 'timeout' or objs[0] == float('inf') or self.check_duplicate_obj(pop, np.round(objs[0], 5)):
            #     continue
            offspring['objective'] = np.round(objs[0], 5)

            return eval_times, pop, offspring
        return eval_times, None, None

    def evolve_algorithm(self, eval_times, pop, node, brother_node, operator):
        for i in range(3):
            eval_times += 1
            _, offspring = self.get_offspring(pop, operator, father=node)
            objs = self.interface_eval.batch_evaluate([offspring['code']], 0)
            # if objs == 'timeout':
            #     return eval_times, None
            # if objs[0] == float('inf') or self.check_duplicate(pop, offspring['code']):
            #     continue
            # offspring['objective'] = np.round(objs[0], 5)

            return eval_times, offspring
        return eval_times, None
