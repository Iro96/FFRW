import random

class MetaController:
    def __init__(self, mutation_prob=0.2):
        self.mutation_prob = mutation_prob

    def decide_mutation(self, student):
        if random.random() < self.mutation_prob:
            student.mutate_add_layer(new_size=random.choice([32,64,128]))
