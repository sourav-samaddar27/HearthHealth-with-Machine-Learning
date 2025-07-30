# HearthHealth-with-Machine-Learning

Heart Health Prediction using Machine Learning
This project focuses on building and evaluating machine learning models to predict heart disease based on various health indicators. It demonstrates a typical machine learning workflow, including data loading, preprocessing, visualization, model training, and evaluation. A key aspect of this project is the implementation of an Epsilon-Greedy algorithm for model selection, inspired by game theory, to dynamically choose between different classification models.

Dataset
The project utilizes the diabetes_012_health_indicators_BRFSS2015.csv dataset, which contains health indicators and diabetes status. For the purpose of this project, the Diabetes_Status column is used as the target variable for predicting "Heart Disease" (renamed from HeartDiseaseorAttack for clarity).

Key Features:
The dataset includes a variety of health and lifestyle indicators such as:

High_BP (High Blood Pressure)

High_Cholesterol

Cholesterol_Check

BMI (Body Mass Index)

Smoker

Stroke

Heart_Disease (Target Variable: 0 for No, 1 for Yes)

Phys_Activity (Physical Activity)

Fruit_Intake

Veg_Intake

Alcohol_Consumption

Any_Healthcare

NoDocbc_Cost (No Doctor because of Cost)

Gen_Health (General Health)

Ment_Health (Mental Health)

Phys_Health (Physical Health)

Walk_Difficulty (Difficulty Walking)

Gender

Age

Education

Income

Key Features & Methodologies
Data Loading & Preprocessing:

Loads data from a CSV file using Pandas.

Renames columns for better readability.

Checks for and handles missing values (imputation using mean strategy for numerical features).

Exploratory Data Analysis (EDA):

Visualizes feature correlations using a heatmap to understand relationships between variables.

Analyzes the distribution of the target variable (Heart_Disease).

Model Selection (Epsilon-Greedy Algorithm):

Implements a game theory-inspired Epsilon-Greedy algorithm to select the best-performing model between LogisticRegression and RandomForestClassifier.

The algorithm balances exploration (trying different models) and exploitation (sticking with the best-performing model so far).

Machine Learning Models:

Logistic Regression: A linear model for binary classification.

Random Forest Classifier: An ensemble learning method for classification.

Model Evaluation:

Calculates Accuracy Score.

Generates a Confusion Matrix.

Provides a Classification Report (Precision, Recall, F1-Score).

Plots the Receiver Operating Characteristic (ROC) Curve and calculates AUC Score.

Model Persistence:

Saves the best-performing model using joblib for future use.

Technologies & Libraries Used
Python

Pandas: For data manipulation and analysis.

NumPy: For numerical operations.

Matplotlib: For data visualization.

Seaborn: For enhanced data visualization.

Scikit-learn: For machine learning models, preprocessing, and evaluation metrics.

train_test_split

accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve

LogisticRegression

RandomForestClassifier

SimpleImputer

Joblib: For saving and loading machine learning models.

Google Colab (for execution environment): The notebook includes from google.colab import files for downloading the model.

⚙️ How to Run the Notebook
Clone the Repository:

git clone <your-repo-url>/HeartHealthPrediction.git
cd HeartHealthPrediction

Download the Dataset:

Ensure you have the diabetes_012_health_indicators_BRFSS2015.csv file in the same directory as the notebook. (You might need to find this dataset online, e.g., on Kaggle).

Open in Jupyter Notebook or Google Colab:

You can run this notebook locally using Jupyter Notebook:

jupyter notebook HeartHealth.ipynb

Alternatively, upload the HeartHealth.ipynb file to Google Colab for a cloud-based execution environment.

Run All Cells: Execute all cells in the notebook sequentially.

Go to Cell > Run All in Jupyter/Colab.

Observe Output:

The notebook will print data previews, missing value checks, model evaluation metrics, and display plots (correlation heatmap, ROC curve).

It will also download the trained model (health_predictor_model.joblib) to your local machine.

Results
The notebook provides an output of the model's performance, including:

Accuracy Score

Confusion Matrix

Classification Report (showing precision, recall, f1-score for each class)

AUC Score

A plot of the ROC Curve.

(Example output from the notebook:)

Best model selected: logistic
Accuracy: 0.9081737346101231



Classification Report:
              precision    recall  f1-score   support

         0.0       0.92      0.99      0.95     10615
         1.0       0.52      0.10      0.17      1081

    accuracy                           0.91     11696
   macro avg       0.72      0.55      0.56     11696
weighted avg       0.88      0.91      0.88     11696


AUC Score: 0.8450391574940426



