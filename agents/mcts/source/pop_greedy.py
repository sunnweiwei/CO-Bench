import heapq

def population_management(pop_input,size):
    pop = [individual for individual in pop_input if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop = [] 
    unique_objectives = []
    for individual in pop:
        if individual['objective'] not in unique_objectives:
            unique_pop.append(individual)
            unique_objectives.append(individual['objective'])
    # Delete the worst individual
    #pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    pop_new = heapq.nsmallest(size, unique_pop, key=lambda x: x['objective'])
    return pop_new

def population_management_s1(pop_input,size):
    pop = [individual for individual in pop_input if individual['objective'] is not None]
    if size > len(pop):
        size = len(pop)
    unique_pop = []
    unique_algorithms = []
    for individual in pop:
        if individual['algorithm'] not in unique_algorithms:
            unique_pop.append(individual)
            unique_algorithms.append(individual['algorithm'])
    # Delete the worst individual
    #pop_new = heapq.nsmallest(size, pop, key=lambda x: x['objective'])
    pop_new = heapq.nlargest(size, unique_pop, key=lambda x: x['objective'])
    return pop_new