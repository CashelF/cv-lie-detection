# Lie Detection using Computer Vision

## Overview
This lie detection system is designed to analyze video inputs to predict deceptive behavior. It utilizes machine learning techniques to assess various physiological and behavioral indicators that correlate with deception.

## Dataset
The dataset consists of 60 videos labeled as truthful and 61 videos labeled as deceptive, primarily in a court setting. The videos are real-life trial data collected at the University of Michigan. The dataset can be found in the folder `Real_Life_Deception/Clips`, categorized into `Truthful` and `Deceptive` subfolders, though we had to ask permission to download the data, so it isn't included in this repo. More details about the dataset are available in the research paper: [Trial.ICMI.pdf](http://web.eecs.umich.edu/~zmohamed/PDFs/Trial.ICMI.pdf).

## Getting Started

### Prerequisites
Before you run the application, ensure that you have Python installed on your system. You can install all the required dependencies from the `requirements.txt` file:

```bash
pip install -r requirements.txt
```

### Running the Application
To start the lie detection application, run the `run.py` script from the command line:

```bash
python run.py
```

This script initializes the system and begins the lie detection process.

### Using a Different Machine Learning Model
The project is configured to use a default machine learning model (logistic regression) with a 62% accuracy on our dataset located in the main directory (`model.joblib`). If you wish to use a different model:

1. Navigate to the `ml_models` folder where alternative machine learning models are provided.
2. Train or select the model you want to use.
3. Use `joblib` to serialize the model:

    ```python
    from joblib import dump
    dump(your_model, 'model.joblib')
    ```

4. Replace the existing `model.joblib` in the main directory with your new model file.

This setup allows you to easily swap between different models depending on your analysis needs or testing scenarios.

## Project Structure
- **run.py**: The main script to start the lie detection system.
- **ml_models/**: Directory containing different machine learning models.
- **image_processing.py**: This file handles image processing for video frames to prepare them for analysis.
- **metrics.py**: Contains the MetricsCalculator class that calculates various metrics needed by the model.
- **config.py**: Stores necessary global values for metrics calculation and other configurations.
- **requirements.txt**: File containing a list of libraries required to run the project.

## Contributing
Contributions to this project are welcome. You can contribute in several ways:
1. Submit bugs and feature requests.
2. Review code and provide feedback.
3. Add or improve documentation.

For major changes, please open an issue first to discuss what you would like to change.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

This project was in part adapted from [Truthsayer](https://github.com/everythingishacked/Truthsayer).
