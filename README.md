# WiDS Datathon++ 2025 University Challenge
#### Unraveling the Mysteries of the Female Brain: Functional Networks Throughout Childhood Development
---
## Overview
- **Kaggle Competition Page:** [WiDS Datathon++ 2025 Challenge](https://www.kaggle.com/competitions/widsdatathon2025-university)
- **Data Provided By:** Healthy Brain Network (HBN), Child Mind Institute (CMI), Reproducible Brain Charts project (RBC).
- **Challenge Overview:** This challenge involves a machine learning task where students will predict age from 2 dimensional functional brain networks (connectomes) derived from fMRI recordings during resting-state.
- **Data:** The HBN dataset includes around 1,578 adolescents aged 5-21, with 63% male and 37% female. Each individual has a functional connectivity network matrix (200x200), and additional information about the individual such as sex, ethnicity, race, etc.
  #### The dataset consists of:
  1. a training folder `train_tsv` consisting of functional connectome matrices for 1,104 individuals
  1. a test folder `test_tsv` consisting of functional connectome matrices for 474 individuals
  1. a metadata folder `metadata` consisting of additional information about individuals in training and test sets
---
## Application Domain
- This challenge employs machine learning and neuroimaging methodologies to investigate brain activity patterns in children and adolescents diagnosed with neuropsychiatric disorders, including anxiety, depression, autism spectrum disorder, and attention-deficit/hyperactivity disorder (ADHD).
- By constructing predictive models that estimate age based on these neural patterns, we aim to identify sex-based differences to facilitate early detection and intervention.

---
## Technologies Used
- **Python:** The primary programming language for data analysis and model development.
- **TensorFlow:** A powerful library used for building and training machine learning models, particularly neural networks.
- **NumPy and Pandas:** Libraries for data manipulation and analysis.
- **scikit-learn:** A library for implementing machine learning algorithms and model evaluation.
 ---
 ## Method Description 
1. **Data Preprocessing:**
   - Extract a row of correlations from the upper triangular portion of the 2-dimensional functional connectivity matrices, creating a vector of correlations for each individual.
   - Perform this operation for the files in both the training and test folders, resulting in the creation of separate training and test dataframes.

2. **Data Concatenation:**
   - Concatenate the training and test dataframes with their respective metadata, integrating demographic and clinical information for further analysis.

3. **Machine Learning Preparation:**
   - The dataset is now ready for machine learning, having transformed the connectome data into a suitable format for model training.

4. **Model Fitting:**
   - Various regression models (e.g., Linear Regression, Random Forest Regression, and Neural Networks) are developed to predict age based on the functional connectivity patterns.
5. **Model Evaluation:**
   - The models are evaluated using Root Mean Squared Error (RMSE) as the primary metric, comparing predicted ages to actual ages from the dataset.
 ---
 ## Team Members
 - **Dascălu Laura-Dumitrina** - Student 
 - **Lupu Gheorghe** - Student
