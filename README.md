# Smart Used Car Price Prediction and Visual Inspection System

## Overview
This project aims to provide a transparent and efficient tool for estimating the market value of used cars and performing visual inspections to detect damages. The system integrates a machine learning pipeline, real-time data collection using Raspberry Pi, and image processing techniques.

## Folder Structure
- `data/`: Contains the dataset file.
- `models/`: Contains the code for data preprocessing, feature engineering, and model training.
- `sensors/`: Contains the code for reading sensor data from the Raspberry Pi.
- `visual_inspection/`: Contains the code for image processing and visual inspection.
- `static/`: Contains static files like CSS.
- `templates/`: Contains HTML templates for the web app.
- `app.py`: The main Flask application file.
- `requirements.txt`: Lists all the dependencies required for the project.
- `README.md`: Provides an overview of the project and instructions for setting up and usage.

## Setup
1. Clone the repository.
2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Place your dataset in the `data/` folder.
4. Run the individual scripts to test each component:
    - `models/model.py`: To train and save the model.
    - `sensors/read_sensors.py`: To read data from sensors.
    - `visual_inspection/detect_damages.py`: To detect damages in car images.
    - `app.py`: To run the complete system.

## Usage
1. Train the machine learning model by running `models/model.py`.
2. Collect real-time data using `sensors/read_sensors.py`.
3. Perform visual inspection using `visual_inspection/detect_damages.py`.
4. Integrate all components and run the complete system using `app.py`.

## Future Work
- Expand the dataset for better model accuracy.
- Enhance real-time data integration.
- Develop a user-friendly web application to make the model accessible to a broader audience.
