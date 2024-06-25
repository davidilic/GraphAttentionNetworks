# Graph Attention Network (GAT) for Node Classification

This project implements a Graph Attention Network (GAT) for node classification on the Cora dataset. The implementation achieves an accuracy of approximately 82% on Cora dataset's test set.

## Overview

Graph Attention Networks (GATs) are a type of graph neural network that use attention mechanisms to improve the process of node feature aggregation. This implementation focuses on node classification in citation networks, specifically using the Cora dataset.

## Model Details

- The GAT uses two graph attention layers.
- The first layer transforms the input features to a hidden representation.
- The second layer produces the final class probabilities.
- Dropout and early stopping are used to prevent overfitting.