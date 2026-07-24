#!/bin/bash
# scripts/aws_setup.sh
# ─────────────────────────────────────────────────────────────
# One-time setup script for your EC2 instance.
# Run this ONCE after launching your EC2 instance.
#
# Tested on: Amazon Linux 2023 / Ubuntu 22.04
#
# Usage:
#   chmod +x scripts/aws_setup.sh
#   ssh -i your-key.pem ec2-user@YOUR_EC2_IP
#   ./aws_setup.sh
# ─────────────────────────────────────────────────────────────

set -e   # exit immediately on any error
echo "======================================================"
echo " EC2 Setup — Cancer Classifier Deployment"
echo "======================================================"

# ── Detect OS ─────────────────────────────────────────────────
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
fi

# ── 1. System Update ─────────────────────────────────────────
echo "[1/6] Updating system packages..."
if [ "$OS" = "amzn" ]; then
    sudo yum update -y
    sudo yum install -y docker git curl unzip
elif [ "$OS" = "ubuntu" ]; then
    sudo apt-get update -y
    sudo apt-get install -y docker.io git curl unzip
fi

# ── 2. Start Docker ───────────────────────────────────────────
echo "[2/6] Starting Docker service..."
sudo systemctl start docker
sudo systemctl enable docker

# Add current user to docker group (avoids sudo on every docker command)
sudo usermod -aG docker $USER
echo "  NOTE: Log out and back in for docker group to take effect"

# ── 3. Install AWS CLI ────────────────────────────────────────
echo "[3/6] Installing AWS CLI v2..."
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
sudo /tmp/aws/install --update
rm -rf /tmp/awscliv2.zip /tmp/aws
aws --version

# ── 4. Configure AWS credentials ─────────────────────────────
echo "[4/6] Configuring AWS credentials..."
echo "  IMPORTANT: For EC2, use an IAM Role instead of hardcoded keys."
echo "  Attach a role with AmazonECR-FullAccess + S3ReadOnlyAccess to this instance."
echo ""
echo "  If you must use keys, run: aws configure"
echo "  (but IAM Role is strongly preferred for security)"

# ── 5. Create ECR Repository (run this from your LOCAL machine) ──
# Uncomment and run from local machine where you have full AWS credentials:
# aws ecr create-repository \
#     --repository-name cancer-classifier \
#     --region us-east-1
#
# Note the repositoryUri from the output — that's your ECR_REGISTRY secret

# ── 6. Setup MLflow (optional — run on this EC2 or separate instance) ──
echo "[5/6] Setting up MLflow tracking server (optional)..."
read -p "  Do you want to run MLflow server on this instance? (y/n): " RUN_MLFLOW

if [ "$RUN_MLFLOW" = "y" ]; then
    pip3 install mlflow boto3 --quiet

    # Run MLflow as a background service
    # Artifacts stored in S3 (replace bucket name)
    echo "  Starting MLflow server..."
    nohup mlflow server \
        --host 0.0.0.0 \
        --port 5000 \
        --backend-store-uri sqlite:///mlflow.db \
        --default-artifact-root s3://your-bucket-name/mlflow-artifacts \
        > /var/log/mlflow.log 2>&1 &

    echo "  MLflow server started at http://$(curl -s ifconfig.me):5000"
    echo "  Add this to your GitHub Secrets as MLFLOW_TRACKING_URI"
fi

# ── 7. Security: Open required ports ─────────────────────────
echo "[6/6] Security Group reminder:"
echo "  Make sure your EC2 Security Group allows inbound:"
echo "    Port 80   (HTTP — FastAPI app)"
echo "    Port 5000 (MLflow UI — restrict to your IP only!)"
echo "    Port 22   (SSH — restrict to your IP only!)"
echo ""
echo "  Configure in AWS Console → EC2 → Security Groups"

echo ""
echo "======================================================"
echo " Setup complete! Next steps:"
echo "======================================================"
echo ""
echo "  1. Attach an IAM Role to this EC2 with ECR + S3 access"
echo "  2. Add GitHub Secrets (see .github/workflows/ci_cd.yml header)"
echo "  3. Push to main branch — GitHub Actions will deploy automatically"
echo ""
echo "  To manually test deployment:"
echo "    docker pull YOUR_ECR_REGISTRY/cancer-classifier:latest"
echo "    docker run -d -p 80:8000 YOUR_ECR_REGISTRY/cancer-classifier:latest"
echo "    curl http://localhost/health"
