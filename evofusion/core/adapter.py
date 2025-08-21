import torch.nn as nn

class KnowledgeAdapter(nn.Module):
    def __init__(self, teacher_dim, student_dim):
        super().__init__()
        self.adapter = nn.Linear(teacher_dim, student_dim)

    def forward(self, teacher_output):
        return self.adapter(teacher_output)
