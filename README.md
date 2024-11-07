# NetIQ - Network Traffic Classifier

NetIQ is a GUI-based application designed to classify network traffic based on applications. It utilizes machine learning algorithms to preprocess data, balance datasets, and perform classification using various models like Random Forest, k-NN, SVM, and Naive Bayes.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Machine Learning Models](#machine-learning-models)
- [Credits](#credits)
- [License](#license)

## Features

- **File Upload:** Allows users to upload CSV files containing network traffic data.
- **Preprocessing and Balancing:** Fills missing values, encodes categorical features, and balances the dataset using RandomUnderSampler.
- **Multiple Classification Algorithms:** Supports Random Forest, k-NN, SVM, and Naive Bayes classifiers.
- **Confusion Matrix Visualization:** Displays confusion matrices for each classification model.

## Installation

Download the executable file from releases

or

1. **Clone the repository:**
    ```bash
    git clone https://github.com/peyushgedela/NetIQ.git
    cd NetIQ
    ```

2. **Create a virtual environment:**
    ```bash
    python -m venv venv
    ```

3. **Activate the virtual environment:**
    - On Windows:
        ```bash
        venv\Scripts\activate
        ```
    - On macOS and Linux:
        ```bash
        source venv/bin/activate
        ```

4. **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1. **Run the application:**
    ```bash
    python gui.py
    ```

2. **Upload CSV File:**
    - Click on the "Upload CSV" button to select and upload your network traffic data file.

3. **Preprocess and Balance:**
    - Click on the "Preprocess and Balance" button to prepare your data for classification.

4. **Classification:**
    - Choose a classification algorithm (Random Forest, k-NN, SVM, Naive Bayes) by clicking the corresponding button.
    - The application will display the accuracy score and confusion matrix for the selected algorithm.

## Machine Learning Models

NetIQ supports the following machine learning models:

- **Random Forest:** An ensemble learning method for classification that operates by constructing multiple decision trees during training.
- **k-NN (k-Nearest Neighbors):** A simple, instance-based learning algorithm where classification is done by identifying the k-nearest neighbors to the input sample.
- **SVM (Support Vector Machine):** A supervised learning model that analyzes data for classification by finding the optimal hyperplane that separates the data into classes.
- **Naive Bayes:** A probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.

## Credits

NetIQ was developed by:
- S Meghanath Reddy [Github Profile](https://github.com/RAF-MAKEOUTHILL)
- K Purnanandh [Github profile](https://github.com/PURNANANDH)
- G Peyush

Special thanks to Prof. L. Anjaneyulu, NIT Warangal for his unending support.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
