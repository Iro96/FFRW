import torch
import torch.nn as nn
from core.adapter import KnowledgeAdapter
from core.meta_controller import MetaController
from core.fitness import fitness_score

def train(student, teacher, data_loader, epochs=10, lr=1e-3):
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    adapter = KnowledgeAdapter(teacher_dim=teacher.output_dim, student_dim=student.output_layer.out_features)
    controller = MetaController()

    for epoch in range(epochs):
        for x, y in data_loader:
            teacher_out = teacher(x).detach()
            student_out = student(x)
            loss = nn.MSELoss()(adapter(teacher_out), student_out)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        controller.decide_mutation(student)
        score = fitness_score(student, loss)
        print(f"Epoch {epoch}, Fitness Score: {score:.4f}")
