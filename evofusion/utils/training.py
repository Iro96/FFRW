import sys
import os

import torch
import torch.nn as nn
from evofusion.core.adapter import KnowledgeAdapter
from evofusion.core.meta_controller import MetaController
from evofusion.core.fitness import fitness_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def train(student, teacher, data_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    adapter = KnowledgeAdapter(teacher_dim=teacher.output_dim, student_dim=student.output_layer.out_features)
    controller = MetaController()

    for epoch in range(epochs):
        total_loss = 0
        for x, y in data_loader:
            teacher_out = teacher(x).detach()
            student_out = student(x)
            loss = nn.MSELoss()(adapter(teacher_out), student_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        controller.decide_mutation(student)
        score = fitness_score(student, loss)
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(data_loader):.4f}, Fitness: {score:.4f}")
