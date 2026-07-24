# 🧬 Personalized Medicine — Cancer Mutation Classifier
> End-to-End MLOps: Training → MLflow → CI/CD → AWS Deployment

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     YOUR LAPTOP                          │
│  src/train.py ──► MLflow (tracks experiments + models)  │
└────────────────────┬────────────────────────────────────┘
                     │ git push
┌────────────────────▼────────────────────────────────────┐
│              GITHUB ACTIONS (CI/CD)                      │
│  1. pytest  2. Train + MLflow  3. Docker build + ECR    │
└────────────────────┬────────────────────────────────────┘
                     │ deploy
┌────────────────────▼────────────────────────────────────┐
│                   AWS (Production)                        │
│  ECR (image registry) ──► EC2 (FastAPI + MLflow model)  │
└─────────────────────────────────────────────────────────┘
```

## 📁 Project Structure

```
personalized-medicine/
├── src/
│   ├── train.py          # Training pipeline with MLflow tracking
│   └── preprocess.py     # Shared text cleaning (used by train + API)
├── app/
│   └── main.py           # FastAPI inference server
├── tests/
│   └── test_pipeline.py  # Unit tests (CI gate)
├── scripts/
│   └── aws_setup.sh      # One-time EC2 setup
├── .github/
│   └── workflows/
│       └── ci_cd.yml     # Full CI/CD pipeline
├── Dockerfile            # Multi-stage production build
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start (Local)

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Download dataset
From [Kaggle](https://www.kaggle.com/c/msk-redefining-cancer-treatment/data):
- `training_variants.csv`
- `training_text.csv`

Place both in `data/`.

### 3. Train with MLflow tracking
```bash
cd src
python train.py --variants ../data/training_variants.csv --text ../data/training_text.csv
```

### 4. View experiments in MLflow UI
```bash
mlflow ui   # opens http://localhost:5000
```

### 5. Run tests
```bash
pytest tests/ -v
```

### 6. Run FastAPI locally
```bash
uvicorn app.main:app --reload --port 8000
# Docs at: http://localhost:8000/docs
```

---

## 🐳 Docker

```bash
# Build
docker build -t cancer-classifier .

# Run
docker run -d \
  -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://your-mlflow-server:5000 \
  --name cancer-classifier \
  cancer-classifier

# Test
curl http://localhost:8000/health
```

---

## ☁️ AWS Setup

### Step 1 — Launch EC2
- AMI: Amazon Linux 2023 or Ubuntu 22.04
- Instance type: `t3.medium` (minimum for this model)
- Storage: 20GB gp3

### Step 2 — Run setup script
```bash
chmod +x scripts/aws_setup.sh
./scripts/aws_setup.sh
```

### Step 3 — Create ECR Repository
```bash
aws ecr create-repository --repository-name cancer-classifier --region us-east-1
```

### Step 4 — Upload data to S3 (one time)
```bash
aws s3 cp data/training_variants.csv s3://your-bucket-name/data/
aws s3 cp data/training_text.csv     s3://your-bucket-name/data/
```

### Step 5 — Add GitHub Secrets
Go to: GitHub → Your Repo → Settings → Secrets and variables → Actions

| Secret | Description |
|--------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user access key |
| `AWS_SECRET_ACCESS_KEY` | IAM user secret key |
| `AWS_REGION` | e.g. `us-east-1` |
| `ECR_REGISTRY` | e.g. `123456789.dkr.ecr.us-east-1.amazonaws.com` |
| `ECR_REPOSITORY` | `cancer-classifier` |
| `EC2_HOST` | Your EC2 public IP |
| `EC2_USER` | `ec2-user` or `ubuntu` |
| `EC2_SSH_KEY` | Full content of your `.pem` file |
| `MLFLOW_TRACKING_URI` | `http://YOUR_EC2_IP:5000` |

### Step 6 — Push to main → CI/CD runs automatically
```bash
git push origin main
```

---

## 🔬 API Usage

### Predict mutation class
```bash
curl -X POST http://YOUR_EC2_IP/predict \
  -H "Content-Type: application/json" \
  -d '{
    "gene": "BRCA1",
    "variation": "R1699Q",
    "clinical_text": "The BRCA1 gene plays a critical role in DNA repair via homologous recombination..."
  }'
```

**Response:**
```json
{
  "predicted_class": 4,
  "predicted_class_name": "Loss-of-function",
  "confidence": 0.7823,
  "all_probabilities": [
    {"class_id": 4, "class_name": "Loss-of-function", "probability": 0.7823},
    {"class_id": 1, "class_name": "Likely Loss-of-function", "probability": 0.1124},
    ...
  ],
  "gene": "BRCA1",
  "variation": "R1699Q"
}
```

### Health check
```bash
curl http://YOUR_EC2_IP/health
```

### Interactive API docs
```
http://YOUR_EC2_IP/docs
```

---

## 📊 MLflow Experiment Tracking

Every training run logs:
- **Parameters**: model type, TF-IDF settings, hyperparameters
- **Metrics**: train log loss, test log loss, overfit gap
- **Artifacts**: confusion matrix plots, classification reports, vectorizers
- **Model**: versioned in the Model Registry

Access MLflow UI: `http://YOUR_MLFLOW_SERVER:5000`

---

## 🧪 CI/CD Flow

```
git push main
     │
     ▼
[pytest]  ──── FAIL ──► ✗ Pipeline stops. No deploy.
     │
   PASS
     │
     ▼
[train.py] → logs to MLflow
     │
     ▼
[docker build] → push to ECR
     │
     ▼
[EC2 deploy] → pull image → restart container → health check
     │
     ▼
  ✓ Live!
```

---

## ⬆️ Next Steps (Level Up)

| What | Why |
|------|-----|
| Replace TF-IDF with BioBERT embeddings | Richer semantic features for clinical text |
| Add class_weight='balanced' to LR | Handle class 7 dominance |
| Add Prometheus + Grafana metrics | Monitor prediction latency and drift |
| Swap EC2 for ECS Fargate | Auto-scaling, no server management |
| Add model drift detection | Retrain automatically when data distribution shifts |
