
# End-to-end Watch Price Prediction Machine Learning Project

# Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Key Features](#-key-features)
- [Steps](#steps)
  - [1. Training Pipeline](#1-training-pipeline)
  - [2. Testing](#2-testing)
  - [3. Model Promotion](#3-model-promotion)
  - [4. Streamlit App and Dockerization](#4-streamlit-app-and-dockerization)
  - [5. Deployment](#5-deployment)
- [Project Prerequisites to run the CI/CD Pipeline🛠️](#project-prerequisites-to-run-the-cicd-pipeline)
- [How to Use This Project](#how-to-use-this-project)
- [What I Learned](#what-i-learned)
- [Contributing](#contributing)


## Project Overview

This machine learning project is a comprehensive end-to-end solution for predicting luxury watch prices using advanced data science and MLOps techniques. The application leverages cutting-edge machine learning models, robust data pipelines, and modern cloud infrastructure to provide accurate watch price predictions.

Check it out on:
http://18.212.223.248:8501/

![image](https://github.com/user-attachments/assets/dd9952c2-226f-4024-892e-6baf77f7a51e)
![image](https://github.com/user-attachments/assets/b40e2a09-1d4d-43f5-b3b9-697c9c0e2416)



## Technologies Used

- Programming Languages: Python

- Libraries: Scikit-learn, CatBoost, Pandas, Streamlit

- Cloud Services: AWS (S3, EC2, ECR)

- MLOps Tools: DVC, MLflow, Git

- CI/CD: GitHub Actions

- Containerization: Docker
## 🚀 Key Features

- _Data Ingestion_: Fetches watch pricing data from AWS S3(Stored there after scraping from a website)

- _Training Pipeline_: Created a model training pipeline that includes dat aingestion, tranformation, training, evaluation and registration(in MLflow) using **DVC**.

- _Machine Learning_: Utilized **CatBoost** as the primary regression model after experimenting with multiple models and  encoding techniques.

- _Version Control_: **Git** and **DVC**

- _Experiment Tracking_: **MLflow** integration on Dagshub

- _Model Serving_: **Streamlit** web application

- _Containerization_: **Docker** image deployment

- _Cloud Infrastructure_: Deployed on **AWS (S3, ECR, EC2)**

- _CI/CD Pipeline_: Automated model training, testing, and deployment using **Github Action**

Here’s a cleaner and more professional markdown version with improved formatting for readability:

---

## Steps

### 1. Training Pipeline

This training pipeline is built using **DVC (Data Version Control)**, a version control system for machine learning projects. DVC enables you to manage datasets, models, and pipelines with Git-like commands, ensuring reproducibility by tracking data and code changes while providing an easily reproducible pipeline structure.

The pipeline consists of the following steps:

 **1.1 Data Ingestion**  

The raw data is fetched from an **AWS S3 bucket** and stored locally in the `artifacts/` directory. This data was originally scraped from a luxury watch merchant website.


**1.2 Data Transformation**  

The raw data, being scraped directly from the web, required extensive cleaning and preprocessing.

- _Challenges:_  
    - Presence of unnecessary features (irrelevant columns).  
    - Missing values in key features.  
    - Inconsistent formatting and noisy entries.  

- _Steps Taken:_ 
    - Dropped irrelevant columns.  
    - Engineered new meaningful features, e.g., extracting **Case Material Coating** from the broader **Case Material** column.  
    - Imputed missing values using various techniques:  
    - **Simple imputation** (mean/median/mode).  
    - **KNN imputation** for structured relationships.  
    - Training a model specifically for imputing complex columns.  
    - Removed features with a significant percentage of missing values.  
    - Split the data into **training** and **test** datasets and saved them in `artifacts/` for subsequent steps.  

    All transformations were guided by insights obtained through detailed **Exploratory Data Analysis (EDA)** ([View EDA Report](#)).


 **1.3 Data Validation**  

Before feeding the data into the model, it was validated to ensure it met the expected structure and schema.

- _Objective:_

    To prevent model training on corrupted or malformed data.

- _Validation Techniques:_
    - Data type checks.  
    - Range checks for numerical features.  
    - Presence of all required columns.  

- _Tools Used:_
    - **Pytest** for automated testing.



**1.4 Model Training**  

The cleaned and validated training data was used to train the model.

- _Model Used:_  
   - **CatBoost**, selected for its efficiency with categorical data.  

- _Pipeline:_
   - Training data was preprocessed (e.g., encoding of categorical variables).  
   - A **Scikit-learn pipeline** was built, including preprocessing and the CatBoost model.  
   - The trained model and its corresponding features were saved as separate pickle files in the `artifacts/` directory.



**1.5 Evaluation**  

The trained model was evaluated on the test dataset.

- _Metrics Calculated:_
    - **Mean Squared Error (MSE)**  
    - **Root Mean Squared Error (RMSE):** Preferred as it penalizes large errors more heavily, making it ideal for price predictions.  
    - **Mean Absolute Error (MAE)**  
    - **R² Score:** Measures the variance explained by the model.  

- _Logging:_
    All metrics, model parameters, and artifacts were logged into **MLflow Experiments** for tracking and comparison.

 **1.6 Model Registration**  
 
After evaluation, the model was registered in **MLflow** with a deployment status of `'staging'`.

**What is MLflow?**  
MLflow is a platform for managing the machine learning lifecycle, offering tools for:  
- **Experiment Tracking:** Tracks metrics, hyperparameters, and artifacts.  
- **Model Registry:** Stores models with metadata and lifecycle states (e.g., staging, production).  

---

### Reproducibility with DVC
The entire pipeline is modular and fully reproducible using DVC commands.  

- **To reproduce the pipeline:**  
   ```bash
   dvc repro
   ```  

**Key Features of the DVC Pipeline:**  
- Modular stages for ingestion, transformation, training, and evaluation.  
- Automatically tracks changes in datasets, models, and code.  
- Ensures consistency in results across different runs.  

--- 

## 2. Testing

-  Model Load Test

    The model load test ensures the machine learning model can be successfully retrieved and loaded from the MLflow registry. This critical validation checks the model's accessibility, verifying that the serialized model artifact is intact, compatible with the current environment, and can be instantiated without any deserialization errors or compatibility issues.

-  Model Signature Test

    The model signature test validates the consistency of the model's input and output schema. It rigorously checks that the input features match the expected type, count, and format used during training, ensuring the model can handle incoming data correctly and maintains its original structural integrity across different execution environments.

- Model Performance Test

    The model performance test is a comparative evaluation that determines whether the candidate model meets predefined performance thresholds and outperforms the existing champion model. By comparing key metrics like Root Mean Squared Error (RMSE), Mean Absolute Error (MAE), and R² Score, this test decides whether the new model is superior enough to progress through the continuous integration and deployment pipeline.
        It compares the candidate model with the champion model if it exists otherwise evaluate the model on certain predefined thresholds.

If the model passes all this tests, it moves forward, otherwise the pipeline stops.

---

## 3.  Model Promotion

If the model passes all the tests, it is ready to become the new crowned champion! The model is promoted as the champion in Mlflow and the previous champion is demoted and archived.

## 4. Streamlit App and Dockerization

To make the trained model accessible, I created a **Streamlit application** and containerized it for deployment. The app predicts luxury watch prices based on user inputs and provides an intuitive interface for non-technical users.

**Streamlit Application**  

The app is designed with the following features:  
- **Input Form:** Users can provide inputs like brand, case material, and dimensions through an easy-to-use interface.  
- **Real-Time Predictions:** Once the input is submitted, the app uses the trained model to generate and display price predictions.  


**Dockerization**  

To ensure the app runs seamlessly across environments, it was containerized into a Docker image named **`watch-price-predictor`**. This allows for consistent deployment across local machines and cloud platforms.

**Steps in Dockerization and Deployment to AWS ECR:**


4.1. **Build the Docker Image:**  

The Docker image is created using a `Dockerfile`, which specifies the environment, dependencies, and app setup.  
```bash
docker build -t watch-price-predictor .
```  
This command packages the application and its dependencies into an image.

4.2. **Tag the Docker Image:**  

Amazon Elastic Container Registry (ECR) is a fully managed container image registry service provided by AWS. To prepare the image for pushing to AWS Elastic Container Registry (ECR), it is tagged with the repository URI. 
```bash
docker tag watch-price-predictor:latest 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest
   ```  
The tag associates the image with the appropriate repository and region in AWS ECR.

4.3. **Push the Docker Image to ECR:**  

The tagged image is pushed to AWS ECR, making it available for deployment.  

```bash
docker push 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest
   ```  
This step uploads the image to the ECR repository for use in cloud-based environments.


**Deployment to EC2**  

Once the image is pushed to AWS ECR, it can be pulled and run on an EC2 instance to host the app. The deployment ensures the app is live and accessible to users, maintaining scalability and reliability.

--- 

## 5. Deployment

The deployment process is handled on an **AWS EC2** instance.

If any old containers are running, they are stopped and removed to avoid conflicts. 
```bash
docker rm -f $( docker ps -a | grep -E "watch-price-predictor|watchApp" | awk '{print $1}') || true
``` 
The EC2 instance logs into AWS Elastic Container Registry (ECR) to access the latest Docker image. 
```bash
 aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 529088280615.dkr.ecr.us-east-1.amazonaws.com
``` 
The new version of the watch-price-predictor image is pulled from the ECR repository. 
```bash
 docker pull 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest
``` 
The new container is launched, and the app is now live!

```bash
 docker run -d -p 8501:8501 --name watchApp-$(date +%Y%m%d-%H%M%S) 529088280615.dkr.ecr.us-east-1.amazonaws.com/watch-price-predictor:latest

``` 

Once this is done, the Streamlit app is accessible, running the latest model and ready to predict luxury watch prices.

---


All these steps are automated using Github Actions.

## Project Prerequisites to run the CI/CD Pipeline🛠️


### AWS Account
- Create an AWS account
- Generate IAM user credentials with appropriate permissions:
  - S3 read/write access
  - ECR repository access
  - EC2 instance management
- Install AWS CLI and configure your credentials if running locally.
```bash
pip install awscli
aws configure
```
 If not running locally, save the aws credentials in your Github Actions secrets.
### Required AWS Services
- **S3**: Data storage(one for raw data storage and second for dvc remote storage)
  ![image](https://github.com/user-attachments/assets/1da2d02c-9a7b-4979-a464-d16c5a03c223)

   <!-- Download the raw_data file from here and upload it on your s3 storage -->
   <!-- https://github.com/abhishekmaher6699/resources/blob/main/watches_final.csv -->

- **ECR**: Docker image repository
- **EC2**: Application deployment
- **IAM**: Access management


### Software Dependencies

- Docker
- Git
- DVC - configured with remote storage
```bash
dvc remote add -d s3-remote s3://your-bucket-name/path/to/folder

```
- Mlflow remote in Dagshub
### GitHub Secrets Configuration
Required GitHub Secrets:
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION`
- `BUCKET_NAME` - bucket where you raw data is saved
- `FILE_KEY` - raw data file
- `MLFLOW_TRACKING_URI` - mlflow uri 
- `MLFLOW_TRACKING_USERNAME` - your dagshub's username
- `MLFLOW_TRACKING_PASSWORD` - your dagshub account's PAT
- `EC2_SSH_PRIVATE_KEY`
- `EC2_PUBLIC_IP`

If you are running locally, just put them in an .env file. 



## How to Use This Project

Here are step-by-step instructions to set up, run, and use this luxury watch price prediction project:

---

### Step 1: Clone the Repository

1. Open a terminal or command prompt.

2. Clone the repository to your local machine:
   ```bash
   git clone <repository-link>
   cd <project-folder>
   ```

---

### Step 2: Set Up the Environment
1. Create a Python virtual environment:
   ```bash
   python -m venv env
   ```
2. Activate the virtual environment:
   - **Linux/macOS**:
     ```bash
     source env/bin/activate
     ```
   - **Windows**:
     ```bash
     env\Scripts\activate
     ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

### Step 3: Configure AWS Credentials
1. If running locally, configure AWS CLI with your credentials:
   ```bash
   aws configure
   ```
2. Alternatively, add the following credentials to a `.env` fileor github secrets:
   ```env
   AWS_ACCESS_KEY_ID=<your-access-key-id>
   AWS_SECRET_ACCESS_KEY=<your-secret-access-key>
   AWS_DEFAULT_REGION=<your-region>
   ```

---

### Step 4: Configure DVC remote
1. Add a DVC remote storage:
   ```bash
   dvc remote add -d s3-remote s3://your-bucket-name/path/to/folder
   ```

---

### Step 5: Run the Training Pipeline
1. Add or update the raw data file in the S3 bucket.
2. Reproduce the entire pipeline:
   ```bash
   dvc repro
   ```

---

### Step 6: Dockerize the app

1. Make sure you have docker installed and set up.

2. Create a docker image and push it to ECR or wherever you want.
   ```bash
   docker build -t watch-price-predictor .
   docker tag watch-price-predictor:latest <ecr-repository-uri>
   docker push <ecr-repository-uri>
   ```

### Step 8: Run the app

If you want to run the container, use the following commands at the location where you want to run it(locally or ec2):

1. Pull and run the Docker container on an EC2 instance:
   ```bash
   docker pull <ecr-repository-uri>
   docker run -d -p 8501:8501 watch-price-predictor
   ```
2. Access the app  at port 8501:
   ```
   http://<ec2-public-ip>:8501 #for EC2
   localhost:8501 #for local
   ```
---


### Step 7: Use the App
1. Provide the following inputs in the app interface:
   - **Brand** (e.g., Rolex, Omega)
   - **Case Material** (e.g., Steel, Gold)
   - **Case Diameter** (e.g., 40mm, 42mm)
   - Additional features as required.
2. Click the **Predict** button to see the estimated watch price.



## What I learned

This project was a significant learning experience, especially in implementing MLOps and working with cloud infrastructure:

- Working with AWS Services: Mastered integration and configuration of AWS S3, EC2, and ECR to enable seamless data management and deployment.
- Dockerization: Containerized the application to ensure consistent deployments across environments.
- Running Tests: Implemented automated tests for data validation, model performance, and deployment using Pytest and CI/CD pipelines.

- Streamlit: Built an intuitive web application for real-time predictions, making the model accessible to non-technical users.
- MLOps and Experiment Tracking: Utilized MLflow for tracking experiments and managing model lifecycle stages (staging, production).
- DVC: Leveraged DVC for version control of datasets, models, and pipeline artifacts, ensuring reproducibility.
- CI/CD: Developed a robust pipeline using GitHub Actions to automate training, testing, and deployment processes.
- End-to-End Project Management: Coordinated data pipelines, ML workflows, and deployment strategies to deliver a production-ready solution.
## Contributing

Contributions are always welcome!

