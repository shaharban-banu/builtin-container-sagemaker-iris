# Iris Classifier — AWS SageMaker Built-in Container Deployment

Trained and deployed an Iris flower classification model as a hosted, real-time inference endpoint on AWS SageMaker, covering the full workflow from training a scikit-learn model to serving predictions via a live endpoint using SageMaker's built-in Scikit-learn container.

## 📌 Overview

This project demonstrates an end-to-end ML deployment pipeline on AWS SageMaker:

1. **Train** a `RandomForestClassifier` on the classic Iris dataset
2. **Package** the trained model artifact
3. **Deploy** it to a SageMaker real-time inference endpoint using the pre-built SKLearn container
4. **Serve predictions** through a custom `inference.py` handler

## 🏗️ Architecture
train.py ──► model.pkl ──► S3 (model.tar.gz) ──► deploy.py ──► SageMaker Endpoint
│
inference.py (handler)
│
Real-time predictions


Install everything with:

```bash
pip install -r requirements.txt
pip install sagemaker boto3
```

## 🚀 How It Works

### 1. Train the model

```bash
python train.py
```

This loads the built-in Iris dataset from scikit-learn, trains a `RandomForestClassifier`, and saves it to `model/model.pkl`.

### 2. Package and upload to S3

The trained model is compressed into `model.tar.gz` and uploaded to an S3 bucket, which `deploy.py` references via `model_data`.

### 3. Deploy to a SageMaker endpoint

```bash
python deploy.py
```

`deploy.py`:
- Creates a `SKLearnModel` pointing to the model artifact in S3 and the `inference.py` entry point
- Checks whether an endpoint with the given name already exists
- Creates a new endpoint, or updates the existing one, accordingly

### 4. Inference handler

`inference.py` defines the four functions SageMaker's SKLearn container calls at serving time:
- `model_fn` — loads the pickled model from the model directory
- `input_fn` — parses incoming JSON request bodies into a NumPy array
- `predict_fn` — runs the model's `.predict()` on the input
- `output_fn` — maps numeric class predictions back to Iris species names (`setosa`, `versicolor`, `virginica`) and returns JSON

## 📊 Model

- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Dataset:** Iris (150 samples, 4 features, 3 classes)
- **Classes:** `setosa`, `versicolor`, `virginica`

## 🧪 Example Prediction Request

Once deployed, the endpoint accepts a JSON array of feature vectors:

```json
[[5.1, 3.5, 1.4, 0.2]]
```

And returns:

```json
{"prediction": ["setosa"]}
```

## ☁️ AWS Services Used

- **Amazon SageMaker** — model deployment and hosting (built-in SKLearn container)
- **Amazon S3** — model artifact storage
- **IAM** — execution role for SageMaker
- **GitHub Actions** — CI/CD automation
