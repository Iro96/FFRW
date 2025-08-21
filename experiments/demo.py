import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from evofusion.core.student import EvoStudent
from evofusion.utils.training import train

# Dummy dataset
X = torch.randn(1000, 10)
Y = torch.randn(1000, 1)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32)

# Teacher & Student
teacher = EvoStudent(10, 1, hidden_layers=[128,128])
student = EvoStudent(10, 1, hidden_layers=[32])

# Train with EvoFusion
train(student, teacher, loader, epochs=20)
exit()