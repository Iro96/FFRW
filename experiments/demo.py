'''
Use this files By enter this command in terminal => $ python -m experiments.{filenames} - Without ".py"
Authors: ZhenYu15, Iro96 
'''

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
from torch.utils.data import DataLoader, TensorDataset
from evofusion.core.student import EvoStudent
from evofusion.core.save_load import load_student
from evofusion.utils.training import train

# Dummy dataset
X = torch.randn(1000, 10)
Y = torch.randn(1000, 1)
dataset = TensorDataset(X, Y)
loader = DataLoader(dataset, batch_size=32)

# Teacher & Student
teacher = EvoStudent(10, 1, hidden_layers=[128,128])
try:
    student = load_student(path="./checkpoints/student.pt", input_dim=10)

except FileNotFoundError:
    student = EvoStudent(10, 1, hidden_layers=[32])

# Train with EvoFusion
train(student, teacher, loader, epochs=20)
exit()