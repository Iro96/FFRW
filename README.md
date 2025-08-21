# EvoFusion: Knowledge Distillation Framework

EvoFusion is a **dynamic, self-evolving knowledge distillation system**. The student network can mutate its architecture during training while learning from any teacher model, regardless of architecture or domain.

---

## Features

- **Self-Evolving Student Network**: Adds layers and neurons dynamically based on training progress.
- **Cross-Domain Knowledge Distillation**: Aligns outputs from heterogeneous teacher models.
- **Function-Preserving Mutations**: Ensures previously learned knowledge is retained.
- **Multi-Objective Fitness Engine**: Balances accuracy, model size, and inference latency.
- **Mutation Monitoring**: Logs architectural changes in real-time.
- **Advanced MetaController**: Guides mutations intelligently based on fitness trends.
- **Checkpoint & Resume**: Save and reload evolving students for long training sessions.

---

## Installation

```bash
git clone https://github.com/Iro96/FFRW.git
cd evofusion
pip install -r requirements.txt
```

## Usage
Run demo version by enter this command in your terminal
```bash
python -m experiments.demo
```

---

## References

- Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- Zoph, B., Le, Q.V. (2017). Neural Architecture Search with Reinforcement Learning.
- Chen, T., Goodfellow, I., Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer.Hinton, G., Vinyals, O., Dean, J. (2015). Distilling the Knowledge in a Neural Network.
- Zoph, B., Le, Q.V. (2017). Neural Architecture Search with Reinforcement Learning.
- Chen, T., Goodfellow, I., Shlens, J. (2016). Net2Net: Accelerating Learning via Knowledge Transfer.
