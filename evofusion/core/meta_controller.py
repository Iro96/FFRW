'''
Updated by ZhenYu15
MetaController is a intelligently guides for student architectural mutations based on fitness and performance trends.
'''

import random

class MetaControllerOld:
    def __init__(self, mutation_prob=0.2):
        self.mutation_prob = mutation_prob

    def decide_mutation(self, student):
        if random.random() < self.mutation_prob:
            student.mutate_add_layer(new_size=random.choice([32,64,128]))

class MetaController:
    def __init__(self, mutation_prob=0.3, fitness_threshold=0.01):
        self.mutation_prob = mutation_prob
        self.fitness_threshold = fitness_threshold
        self.last_fitness = None

    def decide_mutation(self, student, current_fitness):
        # If fitness improvement is small or negative, try mutation
        improve = 0 if self.last_fitness is None else current_fitness - self.last_fitness
        self.last_fitness = current_fitness

        if random.random() < self.mutation_prob or improve < self.fitness_threshold:
            new_size = random.choice([32, 64, 128])
            student.mutate_add_layer(new_size=new_size)
            print(f"[MetaController] Mutation triggered (fitness improvement {improve:.5f})")
            print("-----------------------------------------------------------------------")
