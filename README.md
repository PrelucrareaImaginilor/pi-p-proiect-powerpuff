# WiDS Datathon++ 2025 University Challenge
#### Unraveling the Mysteries of the Female Brain: Functional Networks Throughout Childhood Development
---
## Overview
- **Kaggle Competition Page:** [WiDS Datathon++ 2025 Challenge](https://www.kaggle.com/competitions/widsdatathon2025-university)
- **Data Provided By:** Healthy Brain Network (HBN), Child Mind Institute (CMI), Reproducible Brain Charts project (RBC).
- **Challenge Overview:** This challenge involves a machine learning task where students will predict age from 2 dimensional functional brain networks (connectomes) derived from fMRI recordings during resting-state.
- **Data:** The HBN dataset includes around 1,578 adolescents aged 5-21, with 63% male and 37% female. Each individual has a functional connectivity network matrix (200x200), and additional information about the individual such as sex, ethnicity, race, etc.
  #### The dataset consists of:
  1. a training folder train_tsv consisting of functional connectome matrices for 1,104 individuals
  1. a test folder test_tsv consisting of functional connectome matrices for 474 individuals
  1. a metadata folder metadata consisting of additional information about individuals in training and test sets
---
## Application Domain
- This challenge employs machine learning and neuroimaging methodologies to investigate brain activity patterns in children and adolescents diagnosed with neuropsychiatric disorders, including anxiety, depression, autism spectrum disorder, and attention-deficit/hyperactivity disorder (ADHD).
- By constructing predictive models that estimate age based on these neural patterns, we aim to identify sex-based differences to facilitate early detection and intervention.

---
## Technologies Used
- **Python:** The primary programming language for data analysis and model development.
- **NumPy and Pandas:** Libraries for data manipulation and analysis.
- **scikit-learn:** A library for implementing machine learning algorithms and model evaluation.
 ---
 ## Method Description 
## 1. Data Processing
We extract the upper triangular portion of the two-dimensional functional connectivity matrices for each individual, generating a correlation vector. This represents the functional connections between different brain regions.
This operation is applied to both the training and test datasets.

## 2. Data Preprocessing
### Handling Missing Values (NaN):
At this stage, missing values in both the metadata and the training/test data are handled appropriately.
- **Numerical Variables:** Imputation based on the mean is used, especially for the `bmi` variable (imputation based on sex).
- **Categorical Variables:** For categorical variables (e.g., sex, study_site, ethnicity), missing values are filled with the value "Unknown".

### Categorical Variables Treatment:
One-Hot Encoding is applied to transform categorical variables (e.g., sex, ethnicity, race) into binary variables, eliminating any potential multicollinearity issues.

## 3. Data Concatenation
After extracting the correlation vectors, we integrate these dataframes with their respective metadata. The metadata contains demographic and clinical information such as sex, ethnicity, parental education, etc., which are essential for understanding the variability in connectivity data.
This concatenation creates a complete dataset, including both functional connectivity data and individual features, preparing the data for more detailed analysis.

## 4. Model Training
Before using the data to train machine learning models, we standardize the dataset. This is done using `StandardScaler` to bring all variables to the same scale and prevent variables with larger values from disproportionately influencing the model results.
Once the data is prepared, we develop and train various regression models to predict age based on the functional connectivity data and metadata. The models used include:
- Linear Regression
- Lasso Regression
- Ridge Regression
- ElasticNet Regression

Each of these models will be trained on the training dataset, and predictions will be made on the test dataset.

## 5. Model Evaluation
The developed models will be evaluated using the Root Mean Squared Error (RMSE) metric, which measures the average error between predicted and actual age values.

 ## Team Members
 - **Dascălu Laura-Dumitrina** - Student 
 - **Lupu Gheorghe**
