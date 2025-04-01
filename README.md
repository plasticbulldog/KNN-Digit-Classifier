# KNN Digit Classifier

This project uses the **K-Nearest Neighbors (KNN)** algorithm to classify handwritten digits using the `sklearn` digits dataset.

## How it Works

- Loads a dataset of 8x8 handwritten digits.
- Splits the data into training and test sets.
- Uses KNN with k=3 to predict digit classes.
- Displays the prediction results and test accuracy.

## Getting Started

1. Install dependencies:
```bash
pip install scikit-learn matplotlib
```

2. Run the script:
```bash
python knn_digit_classifier.py
```

## Output

The model achieves around **98% accuracy** on the test set and shows a few visual predictions.

## Hardware Extension Idea

Use a Raspberry Pi with a camera to capture handwritten digits in real life. Preprocess and classify them using this trained KNN model to build an interactive digit recognition system.

---

Author: [plasticbulldog](https://github.com/plasticbulldog)
