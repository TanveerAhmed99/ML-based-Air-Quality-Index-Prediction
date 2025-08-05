ML-based Air Quality Index Prediction
This repository contains a machine learning pipeline built to predict the Air Quality Index (AQI) using real-world environmental data. The project uses a combination of preprocessing techniques, exploratory data analysis (EDA), and various ML models such as XGBoost, CatBoost, and TensorFlow-based neural networks to deliver accurate AQI predictions.

📊 Project Overview
Air pollution is a major environmental health threat. Predicting the AQI helps policymakers and citizens take timely actions to avoid harmful exposure. This project processes an Excel dataset with multiple sheets from different monitoring stations and applies machine learning to model the AQI based on various atmospheric conditions.

🔍 Features
Read and combine AQI data from multiple Excel sheets

Handle missing data and outliers

Visualize distribution and correlations among features

Train and evaluate multiple ML models:

XGBoost

CatBoost

TensorFlow DNN

Compare model performance using metrics like RMSE, MAE, R²

🧪 Technologies Used
Python

Pandas, NumPy for data manipulation

Matplotlib, Seaborn for visualization

Scikit-learn for preprocessing and evaluation

XGBoost, CatBoost, TensorFlow for modeling

Google Colab for cloud-based development

📁 File Structure
ML-based-Air-Quality-Index-Prediction.ipynb: Main Jupyter notebook containing all code

air_quality_index_dataset.xlsx: Dataset used (loaded from Google Drive)

🚀 How to Run
Clone this repository:

bash
Copy
Edit
git clone https://github.com/yourusername/air-quality-index-prediction.git
cd air-quality-index-prediction
Open the notebook in Google Colab or Jupyter and ensure required packages are installed:

python
Copy
Edit
!pip install xgboost catboost tensorflow
Mount Google Drive (if using Colab) and provide the correct path to the dataset:

python
Copy
Edit
from google.colab import drive
drive.mount('/content/drive')
Run all cells in sequence to train and evaluate models.

📈 Results
Visual plots and evaluation metrics help identify the most effective model

Models achieve strong prediction accuracy with low error margins

🤝 Contributing
Contributions are welcome! If you have suggestions or improvements, feel free to fork the repo and submit a pull request.

📄 License
This project is open-source and available under the MIT License.

