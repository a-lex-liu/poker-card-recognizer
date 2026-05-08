## Demo

### Input
![input](demo.jpg)

### Output
![output](results_demo.jpg)

## Model Comparison
| Model | Precision | Recall | F1 Score |
|---|---|---|---|
| 10 Epochs | 0.8665 | 0.9902 | 0.9242 |
| 50 Epochs | 0.9672 | 0.9972 | 0.9820 |

Training for 50 epochs significantly reduced false positives:

- 10 epochs: 1158 false positives
- 50 epochs: 257 false positives

## Failure Cases

The dataset labels only the top-left and bottom-right card indices rather than the entire card body.

As a result, the model occasionally predicts additional detections on the top-right or bottom-left regions of symmetric cards. These detections are counted as false positives during evaluation even though the visual patterns are very similar to the labeled regions.

### Example Failure

![failure1](failure1.jpg)

![failure2](failure2.jpg)

![failure3](failure3.jpg)