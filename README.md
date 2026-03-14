# GNN for Noisy Node Classification

This project explores **Graph Neural Networks (GNNs)** for node classification on a graph with partially labeled data and noisy labels. The goal is to predict labels for unlabeled nodes using graph structure and node features.

The project implements multiple GNN architectures and compares their performance on the dataset.

---

## Project Structure

```
gnn-noisy-node-classification
│
├── data
│   ├── edges.csv
│   ├── node_features.csv
│   ├── train_labels.csv
│   ├── unlabeled_pool.csv
│   └── sample_submission.csv
│
├── src
│   ├── load_data.py          # Loads graph data and creates PyG Data object
│   ├── models.py             # GCN, GAT, GraphSAGE implementations
│   ├── train.py              # Training loop and loss computation
│   └── pseudo_labels.py      # Semi-supervised pseudo-labeling
│
├── main.py                   # Main pipeline script
└── README.md
```

---

## Models Implemented

The following Graph Neural Network architectures are implemented:

* **GCN (Graph Convolutional Network)**
* **GAT (Graph Attention Network)**
* **GraphSAGE**

These models use graph structure (`edges.csv`) and node features (`node_features.csv`) to perform node classification.

---

## Semi-Supervised Learning

The project also includes **pseudo-labeling**:

1. Train a model on labeled nodes.
2. Predict labels for unlabeled nodes.
3. Select predictions with high confidence.
4. Add them to the training set.
5. Retrain the model.

This allows the model to **leverage unlabeled nodes** in the graph.

---

## Installation

Install dependencies:

```bash
pip install torch
pip install torch-geometric
pip install pandas
pip install scikit-learn
```

---

## Running the Project

Run the training pipeline:

```bash
python main.py
```

This will:

1. Load the graph dataset
2. Train the selected GNN model
3. Generate predictions
4. Create a submission file

---

## Dataset

The dataset contains:

| File                  | Description                  |
| --------------------- | ---------------------------- |
| edges.csv             | Graph connectivity           |
| node_features.csv     | Feature vector for each node |
| train_labels.csv      | Labels for training nodes    |
| unlabeled_pool.csv    | Nodes without labels         |
| sample_submission.csv | Format for predictions       |

---

## Output

The models generate prediction files such as:

```
gcn_attempt.csv
gat_attempt.csv
sage_attempt.csv
sage_semi_supervision_submission.csv
```

---

## Technologies Used

* Python
* PyTorch
* PyTorch Geometric
* Scikit-learn
* Pandas

---

## Author

GitHub: https://github.com/sh051
