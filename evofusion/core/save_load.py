import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from evofusion.core.student import EvoStudent

def save_student(student, path="student.pt"):
    torch.save({
        "model_state": student.state_dict(),
        "output_dim": student.output_dim,
        "hidden_layers": [layer.out_features for layer in student.hidden_layers]
    }, path)
    print(f"[Save] Student model saved to {path}")

def load_student(path="student.pt", input_dim=10):
    checkpoint = torch.load(path)
    hidden_layers = checkpoint["hidden_layers"]
    output_dim = checkpoint["output_dim"]

    student = EvoStudent(input_dim=input_dim, output_dim=output_dim, hidden_layers=hidden_layers)
    student.load_state_dict(checkpoint["model_state"])
    print(f"[Load] Student model loaded from {path}")
    return student
