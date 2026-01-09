# ReasonKit Core Deployment Guide

> Production-ready deployment strategies for ReasonKit Core
> Version: 1.0.0 | Last Updated: 2026-01-01

---

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Docker Deployment](#docker-deployment)
3. [Docker Compose Deployment](#docker-compose-deployment)
4. [Kubernetes Deployment](#kubernetes-deployment)
5. [Proxmox LXC Deployment](#proxmox-lxc-deployment)
6. [Cloud Provider Guides](#cloud-provider-guides)
   - [AWS](#aws-deployment)
   - [Google Cloud Platform](#gcp-deployment)
   - [Microsoft Azure](#azure-deployment)
7. [Environment Configuration](#environment-configuration)
8. [Health Checks](#health-checks)
9. [Scaling Considerations](#scaling-considerations)
10. [Security Hardening](#security-hardening)
11. [Monitoring & Observability](#monitoring--observability)
12. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements

| Component | Minimum   | Recommended | Production   |
| --------- | --------- | ----------- | ------------ |
| CPU       | 2 cores   | 4 cores     | 8+ cores     |
| RAM       | 2 GB      | 4 GB        | 8+ GB        |
| Storage   | 10 GB SSD | 50 GB SSD   | 100+ GB NVMe |
| Network   | 100 Mbps  | 1 Gbps      | 10 Gbps      |

### Software Requirements

- Docker 24.0+ (for container deployments)
- Kubernetes 1.28+ (for K8s deployments)
- Rust 1.74+ (for source builds)

### Required API Keys

Configure these via environment variables (NEVER hardcode):

```bash
# LLM Providers (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-...

# Optional: Vector Database
QDRANT_API_KEY=...
QDRANT_URL=http://localhost:6333

# Optional: Redis Cache
REDIS_URL=redis://localhost:6379
```

---

## Docker Deployment

### Quick Start

```bash
# Pull the official image
docker pull reasonkit/core:latest

# Run with basic configuration
docker run -d \
  --name reasonkit-core \
  -p 9100:9100 \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  -e RUST_LOG=info \
  -v reasonkit-data:/app/data \
  reasonkit/core:latest serve-mcp --port 9100
```

### Build from Source

```bash
# Clone repository
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core

# Build the image
docker build -t reasonkit/core:local .

# Run locally built image
docker run -d \
  --name reasonkit-core \
  -p 9100:9100 \
  -e ANTHROPIC_API_KEY="${ANTHROPIC_API_KEY}" \
  reasonkit/core:local serve-mcp --port 9100
```

### Multi-Architecture Build

```bash
# Build for multiple platforms
docker buildx build \
  --platform linux/amd64,linux/arm64 \
  -t reasonkit/core:latest \
  --push .
```

### Production Dockerfile

The included `Dockerfile` uses multi-stage builds for minimal image size:

```dockerfile
# Stage 1: Build (rust:1-slim-bookworm)
# - Compiles with --release
# - Strips binary for size reduction

# Stage 2: Runtime (debian:bookworm-slim)
# - Minimal runtime dependencies
# - Non-root user (reasonkit:1000)
# - Health check included
```

---

## Docker Compose Deployment

### Minimal Deployment

Create `docker-compose.yml`:

```yaml
version: "3.8"

services:
  reasonkit-core:
    image: reasonkit/core:latest
    container_name: reasonkit-core
    restart: unless-stopped
    ports:
      - "9100:9100"
    environment:
      - RUST_LOG=info
      - REASONKIT_ENV=production
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
    volumes:
      - reasonkit-data:/app/data
    healthcheck:
      test: ["CMD", "rk", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s

volumes:
  reasonkit-data:
    driver: local
```

### Full Production Stack

Create `docker-compose.production.yml`:

```yaml
version: "3.8"

networks:
  reasonkit-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16
  monitoring-network:
    driver: bridge

volumes:
  reasonkit-data:
  qdrant-data:
  redis-data:
  prometheus-data:
  grafana-data:
  loki-data:

services:
  # ===========================================================================
  # CORE APPLICATION
  # ===========================================================================
  reasonkit-core:
    image: reasonkit/core:latest
    container_name: reasonkit-core
    restart: unless-stopped
    ports:
      - "9100:9100"
    networks:
      - reasonkit-network
      - monitoring-network
    volumes:
      - reasonkit-data:/app/data
      - ./config:/app/config:ro
    environment:
      # Application
      - RUST_LOG=info
      - REASONKIT_ENV=production
      # API Keys (from .env)
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - OPENAI_API_KEY=${OPENAI_API_KEY}
      - OPENROUTER_API_KEY=${OPENROUTER_API_KEY}
      # Database connections
      - QDRANT_URL=http://qdrant:6333
      - REDIS_URL=redis://redis:6379
      # Server
      - SERVER_HOST=0.0.0.0
      - SERVER_PORT=8080
      - SERVER_WORKERS=4
      # Features
      - ENABLE_METRICS=true
      - ENABLE_TRACING=true
    depends_on:
      qdrant:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "rk", "--version"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "1.0"
          memory: 2G
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp

  # ===========================================================================
  # VECTOR DATABASE
  # ===========================================================================
  qdrant:
    image: qdrant/qdrant:v1.12.5
    container_name: reasonkit-qdrant
    restart: unless-stopped
    ports:
      - "6333:6333"
      - "6334:6334"
    networks:
      - reasonkit-network
    volumes:
      - qdrant-data:/qdrant/storage
    environment:
      - QDRANT__SERVICE__HTTP_PORT=6333
      - QDRANT__SERVICE__GRPC_PORT=6334
      - QDRANT__LOG_LEVEL=INFO
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s
    deploy:
      resources:
        limits:
          cpus: "2.0"
          memory: 4G
        reservations:
          cpus: "0.5"
          memory: 1G

  # ===========================================================================
  # CACHE LAYER
  # ===========================================================================
  redis:
    image: redis:7-alpine
    container_name: reasonkit-redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    networks:
      - reasonkit-network
    volumes:
      - redis-data:/data
    command: >
      redis-server
      --appendonly yes
      --maxmemory 512mb
      --maxmemory-policy allkeys-lru
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 10s
    deploy:
      resources:
        limits:
          cpus: "1.0"
          memory: 1G
        reservations:
          cpus: "0.25"
          memory: 256M

  # ===========================================================================
  # REVERSE PROXY
  # ===========================================================================
  nginx:
    image: nginx:1.27-alpine
    container_name: reasonkit-nginx
    restart: unless-stopped
    ports:
      - "80:80"
      - "443:443"
    networks:
      - reasonkit-network
    volumes:
      - ./nginx/nginx.conf:/etc/nginx/nginx.conf:ro
      - ./nginx/ssl:/etc/nginx/ssl:ro
    depends_on:
      - reasonkit-core
    healthcheck:
      test: ["CMD", "nginx", "-t"]
      interval: 30s
      timeout: 10s
      retries: 3

  # ===========================================================================
  # MONITORING STACK
  # ===========================================================================
  prometheus:
    image: prom/prometheus:v3.0.0
    container_name: reasonkit-prometheus
    restart: unless-stopped
    ports:
      - "9090:9090"
    networks:
      - monitoring-network
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - "--config.file=/etc/prometheus/prometheus.yml"
      - "--storage.tsdb.path=/prometheus"
      - "--storage.tsdb.retention.time=30d"
      - "--web.enable-lifecycle"
    healthcheck:
      test: ["CMD", "wget", "--spider", "-q", "http://localhost:9090/-/healthy"]
      interval: 30s
      timeout: 10s
      retries: 3

  grafana:
    image: grafana/grafana:11.4.0
    container_name: reasonkit-grafana
    restart: unless-stopped
    ports:
      - "3000:3000"
    networks:
      - monitoring-network
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards:ro
      - ./monitoring/grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml:ro
    environment:
      - GF_SECURITY_ADMIN_USER=${GRAFANA_ADMIN_USER:-admin}
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_ADMIN_PASSWORD}
      - GF_USERS_ALLOW_SIGN_UP=false
    depends_on:
      - prometheus

  loki:
    image: grafana/loki:3.0.0
    container_name: reasonkit-loki
    restart: unless-stopped
    ports:
      - "3100:3100"
    networks:
      - monitoring-network
    volumes:
      - loki-data:/loki
      - ./monitoring/loki-config.yml:/etc/loki/local-config.yaml:ro
    command: -config.file=/etc/loki/local-config.yaml
```

### Environment File (.env)

```bash
# .env - DO NOT COMMIT TO VERSION CONTROL
# ReasonKit Production Environment Variables

# =============================================================================
# LLM PROVIDER API KEYS
# =============================================================================
ANTHROPIC_API_KEY=sk-ant-api03-...
OPENAI_API_KEY=sk-...
OPENROUTER_API_KEY=sk-or-v1-...

# =============================================================================
# DATABASE CONFIGURATION
# =============================================================================
QDRANT_URL=http://qdrant:6333
REDIS_URL=redis://redis:6379

# =============================================================================
# MONITORING
# =============================================================================
GRAFANA_ADMIN_USER=admin
GRAFANA_ADMIN_PASSWORD=CHANGE_ME_IN_PRODUCTION

# =============================================================================
# APPLICATION
# =============================================================================
REASONKIT_ENV=production
RUST_LOG=info
SERVER_WORKERS=4
```

### Starting the Stack

```bash
# Create .env file with your API keys
cp .env.example .env
vim .env

# Start all services
docker compose -f docker-compose.production.yml up -d

# View logs
docker compose -f docker-compose.production.yml logs -f reasonkit-core

# Stop all services
docker compose -f docker-compose.production.yml down

# Stop and remove volumes (CAUTION: data loss)
docker compose -f docker-compose.production.yml down -v
```

---

## Kubernetes Deployment

### Namespace and ConfigMap

```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: reasonkit
  labels:
    app.kubernetes.io/name: reasonkit
    app.kubernetes.io/part-of: reasonkit
---
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: reasonkit-config
  namespace: reasonkit
data:
  RUST_LOG: "info"
  REASONKIT_ENV: "production"
  SERVER_HOST: "0.0.0.0"
  SERVER_PORT: "8080"
  SERVER_WORKERS: "4"
  ENABLE_METRICS: "true"
  ENABLE_TRACING: "true"
```

### Secrets

```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: reasonkit-secrets
  namespace: reasonkit
type: Opaque
stringData:
  ANTHROPIC_API_KEY: "sk-ant-api03-..."
  OPENAI_API_KEY: "sk-..."
  OPENROUTER_API_KEY: "sk-or-v1-..."
  QDRANT_API_KEY: ""
```

### Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: reasonkit-core
  namespace: reasonkit
  labels:
    app: reasonkit-core
    app.kubernetes.io/name: reasonkit-core
    app.kubernetes.io/version: "0.1.0"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: reasonkit-core
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxSurge: 1
      maxUnavailable: 0
  template:
    metadata:
      labels:
        app: reasonkit-core
      annotations:
        prometheus.io/scrape: "true"
        prometheus.io/port: "8080"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: reasonkit
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
        runAsGroup: 1000
        fsGroup: 1000
      containers:
        - name: reasonkit-core
          image: reasonkit/core:latest
          imagePullPolicy: Always
          ports:
            - name: http
              containerPort: 8080
              protocol: TCP
          envFrom:
            - configMapRef:
                name: reasonkit-config
            - secretRef:
                name: reasonkit-secrets
          env:
            - name: QDRANT_URL
              value: "http://qdrant.reasonkit.svc.cluster.local:6333"
            - name: REDIS_URL
              value: "redis://redis.reasonkit.svc.cluster.local:6379"
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2000m"
              memory: "4Gi"
          livenessProbe:
            exec:
              command:
                - rk
                - --version
            initialDelaySeconds: 10
            periodSeconds: 30
            timeoutSeconds: 5
            failureThreshold: 3
          readinessProbe:
            httpGet:
              path: /health
              port: http
            initialDelaySeconds: 5
            periodSeconds: 10
            timeoutSeconds: 3
            failureThreshold: 3
          volumeMounts:
            - name: data
              mountPath: /app/data
            - name: config
              mountPath: /app/config
              readOnly: true
          securityContext:
            allowPrivilegeEscalation: false
            readOnlyRootFilesystem: true
            capabilities:
              drop:
                - ALL
      volumes:
        - name: data
          persistentVolumeClaim:
            claimName: reasonkit-data
        - name: config
          configMap:
            name: reasonkit-config
      affinity:
        podAntiAffinity:
          preferredDuringSchedulingIgnoredDuringExecution:
            - weight: 100
              podAffinityTerm:
                labelSelector:
                  matchLabels:
                    app: reasonkit-core
                topologyKey: kubernetes.io/hostname
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: ScheduleAnyway
          labelSelector:
            matchLabels:
              app: reasonkit-core
```

### Service

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: reasonkit-core
  namespace: reasonkit
  labels:
    app: reasonkit-core
spec:
  type: ClusterIP
  ports:
    - port: 8080
      targetPort: http
      protocol: TCP
      name: http
  selector:
    app: reasonkit-core
```

### Ingress

```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: reasonkit-ingress
  namespace: reasonkit
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"
    nginx.ingress.kubernetes.io/proxy-read-timeout: "300"
    nginx.ingress.kubernetes.io/proxy-send-timeout: "300"
spec:
  tls:
    - hosts:
        - api.reasonkit.sh
      secretName: reasonkit-tls
  rules:
    - host: api.reasonkit.sh
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: reasonkit-core
                port:
                  number: 8080
```

### Horizontal Pod Autoscaler

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reasonkit-core-hpa
  namespace: reasonkit
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reasonkit-core
  minReplicas: 3
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
      policies:
        - type: Percent
          value: 10
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 0
      policies:
        - type: Percent
          value: 100
          periodSeconds: 15
        - type: Pods
          value: 4
          periodSeconds: 15
      selectPolicy: Max
```

### PersistentVolumeClaim

```yaml
# pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: reasonkit-data
  namespace: reasonkit
spec:
  accessModes:
    - ReadWriteOnce
  storageClassName: fast-ssd
  resources:
    requests:
      storage: 50Gi
```

### Apply Kubernetes Manifests

```bash
# Apply all manifests
kubectl apply -f namespace.yaml
kubectl apply -f configmap.yaml
kubectl apply -f secrets.yaml
kubectl apply -f pvc.yaml
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml
kubectl apply -f ingress.yaml
kubectl apply -f hpa.yaml

# Or use kustomization
kubectl apply -k ./k8s/

# Check status
kubectl get pods -n reasonkit
kubectl get svc -n reasonkit
kubectl get hpa -n reasonkit

# View logs
kubectl logs -f deployment/reasonkit-core -n reasonkit

# Port forward for local testing
kubectl port-forward svc/reasonkit-core 9100:9100 -n reasonkit
```

### Helm Chart (Optional)

For complex deployments, use the Helm chart:

```bash
# Add ReasonKit Helm repository
helm repo add reasonkit https://charts.reasonkit.sh
helm repo update

# Install with default values
helm install reasonkit reasonkit/reasonkit-core \
  --namespace reasonkit \
  --create-namespace

# Install with custom values
helm install reasonkit reasonkit/reasonkit-core \
  --namespace reasonkit \
  --create-namespace \
  --set replicaCount=5 \
  --set resources.limits.memory=8Gi \
  --set-file secrets.anthropicApiKey=./anthropic-key.txt

# Upgrade
helm upgrade reasonkit reasonkit/reasonkit-core \
  --namespace reasonkit \
  --reuse-values \
  --set image.tag=v0.2.0
```

---

## Proxmox LXC Deployment

Proxmox LXC containers provide near-native performance with lightweight isolation. This is ideal for homelab and edge deployments.

### Create LXC Container

```bash
# On Proxmox host
# Download Debian 12 template if not present
pveam update
pveam download local debian-12-standard_12.2-1_amd64.tar.zst

# Create container
pct create 200 local:vztmpl/debian-12-standard_12.2-1_amd64.tar.zst \
  --hostname reasonkit \
  --cores 4 \
  --memory 4096 \
  --swap 1024 \
  --storage local-lvm \
  --rootfs local-lvm:50 \
  --net0 name=eth0,bridge=vmbr0,ip=dhcp \
  --unprivileged 1 \
  --features nesting=1 \
  --start 1

# Enter container
pct enter 200
```

### Install Dependencies

```bash
# Update system
apt update && apt upgrade -y

# Install build dependencies
apt install -y \
  curl \
  git \
  build-essential \
  pkg-config \
  libssl-dev \
  ca-certificates

# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source ~/.cargo/env

# Verify installation
rustc --version
cargo --version
```

### Install ReasonKit

```bash
# Option 1: Install from crates.io
cargo install reasonkit-core

# Option 2: Build from source
git clone https://github.com/reasonkit/reasonkit-core.git
cd reasonkit-core
cargo build --release
cp target/release/rk /usr/local/bin/

# Verify installation
rk --version
```

### Create Systemd Service

```bash
# Create service user
useradd -r -s /bin/false -d /opt/reasonkit reasonkit

# Create directories
mkdir -p /opt/reasonkit/{data,config,logs}
chown -R reasonkit:reasonkit /opt/reasonkit

# Create environment file
cat > /opt/reasonkit/.env << 'EOF'
RUST_LOG=info
REASONKIT_ENV=production
ANTHROPIC_API_KEY=sk-ant-api03-...
SERVER_HOST=0.0.0.0
SERVER_PORT=8080
EOF
chmod 600 /opt/reasonkit/.env
chown reasonkit:reasonkit /opt/reasonkit/.env

# Create systemd service
cat > /etc/systemd/system/reasonkit.service << 'EOF'
[Unit]
Description=ReasonKit Core - AI Reasoning Infrastructure
Documentation=https://reasonkit.sh/docs
After=network.target

[Service]
Type=simple
User=reasonkit
Group=reasonkit
WorkingDirectory=/opt/reasonkit
EnvironmentFile=/opt/reasonkit/.env
ExecStart=/usr/local/bin/rk serve-mcp --port 8080
ExecReload=/bin/kill -HUP $MAINPID
Restart=always
RestartSec=5
StandardOutput=append:/opt/reasonkit/logs/reasonkit.log
StandardError=append:/opt/reasonkit/logs/reasonkit.log

# Security hardening
NoNewPrivileges=yes
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
PrivateDevices=yes
ProtectKernelTunables=yes
ProtectKernelModules=yes
ProtectControlGroups=yes
ReadWritePaths=/opt/reasonkit/data /opt/reasonkit/logs

# Resource limits
LimitNOFILE=65535
LimitNPROC=4096
MemoryMax=4G
CPUQuota=200%

[Install]
WantedBy=multi-user.target
EOF

# Enable and start service
systemctl daemon-reload
systemctl enable reasonkit
systemctl start reasonkit
systemctl status reasonkit
```

### Configure Firewall

```bash
# Install ufw
apt install -y ufw

# Configure rules
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 8080/tcp comment 'ReasonKit API'
ufw enable

# Verify
ufw status verbose
```

### Log Rotation

```bash
# Create logrotate configuration
cat > /etc/logrotate.d/reasonkit << 'EOF'
/opt/reasonkit/logs/*.log {
    daily
    rotate 14
    compress
    delaycompress
    missingok
    notifempty
    create 0640 reasonkit reasonkit
    postrotate
        systemctl reload reasonkit > /dev/null 2>&1 || true
    endscript
}
EOF
```

### Proxmox Backup

```bash
# Create backup from Proxmox host
vzdump 200 --storage local --compress zstd --mode snapshot

# Schedule automated backups (add to /etc/cron.d/vzdump)
# 0 2 * * * root vzdump 200 --storage backup-storage --compress zstd --mode snapshot --mailto admin@example.com
```

---

## Cloud Provider Guides

### AWS Deployment

#### ECS Fargate

```bash
# Create ECR repository
aws ecr create-repository --repository-name reasonkit/core

# Authenticate Docker
aws ecr get-login-password --region us-east-1 | \
  docker login --username AWS --password-stdin ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com

# Tag and push image
docker tag reasonkit/core:latest ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/reasonkit/core:latest
docker push ${AWS_ACCOUNT_ID}.dkr.ecr.us-east-1.amazonaws.com/reasonkit/core:latest
```

**Task Definition (task-definition.json):**

```json
{
  "family": "reasonkit-core",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole",
  "taskRoleArn": "arn:aws:iam::ACCOUNT_ID:role/reasonkitTaskRole",
  "containerDefinitions": [
    {
      "name": "reasonkit-core",
      "image": "ACCOUNT_ID.dkr.ecr.us-east-1.amazonaws.com/reasonkit/core:latest",
      "essential": true,
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        { "name": "RUST_LOG", "value": "info" },
        { "name": "REASONKIT_ENV", "value": "production" }
      ],
      "secrets": [
        {
          "name": "ANTHROPIC_API_KEY",
          "valueFrom": "arn:aws:secretsmanager:us-east-1:ACCOUNT_ID:secret:reasonkit/api-keys:ANTHROPIC_API_KEY::"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/reasonkit-core",
          "awslogs-region": "us-east-1",
          "awslogs-stream-prefix": "ecs"
        }
      },
      "healthCheck": {
        "command": ["CMD-SHELL", "rk --version || exit 1"],
        "interval": 30,
        "timeout": 5,
        "retries": 3,
        "startPeriod": 60
      }
    }
  ]
}
```

```bash
# Register task definition
aws ecs register-task-definition --cli-input-json file://task-definition.json

# Create service
aws ecs create-service \
  --cluster reasonkit-cluster \
  --service-name reasonkit-core \
  --task-definition reasonkit-core \
  --desired-count 3 \
  --launch-type FARGATE \
  --network-configuration "awsvpcConfiguration={subnets=[subnet-xxx],securityGroups=[sg-xxx],assignPublicIp=ENABLED}" \
  --load-balancers "targetGroupArn=arn:aws:elasticloadbalancing:...,containerName=reasonkit-core,containerPort=8080"
```

#### EKS (Elastic Kubernetes Service)

```bash
# Create EKS cluster
eksctl create cluster \
  --name reasonkit-cluster \
  --region us-east-1 \
  --nodegroup-name standard-workers \
  --node-type t3.large \
  --nodes 3 \
  --nodes-min 2 \
  --nodes-max 10 \
  --managed

# Install AWS Load Balancer Controller
helm repo add eks https://aws.github.io/eks-charts
helm install aws-load-balancer-controller eks/aws-load-balancer-controller \
  -n kube-system \
  --set clusterName=reasonkit-cluster

# Apply Kubernetes manifests (see Kubernetes section)
kubectl apply -k ./k8s/overlays/aws/
```

#### Terraform (AWS)

```hcl
# main.tf
terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "reasonkit-vpc"
  cidr = "10.0.0.0/16"

  azs             = ["${var.aws_region}a", "${var.aws_region}b", "${var.aws_region}c"]
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]

  enable_nat_gateway = true
  single_nat_gateway = false

  tags = {
    Environment = "production"
    Project     = "reasonkit"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "reasonkit" {
  name = "reasonkit-cluster"

  setting {
    name  = "containerInsights"
    value = "enabled"
  }

  tags = {
    Environment = "production"
    Project     = "reasonkit"
  }
}

# Secrets Manager
resource "aws_secretsmanager_secret" "api_keys" {
  name = "reasonkit/api-keys"
}

resource "aws_secretsmanager_secret_version" "api_keys" {
  secret_id = aws_secretsmanager_secret.api_keys.id
  secret_string = jsonencode({
    ANTHROPIC_API_KEY  = var.anthropic_api_key
    OPENAI_API_KEY     = var.openai_api_key
    OPENROUTER_API_KEY = var.openrouter_api_key
  })
}

# Application Load Balancer
module "alb" {
  source  = "terraform-aws-modules/alb/aws"
  version = "~> 9.0"

  name    = "reasonkit-alb"
  vpc_id  = module.vpc.vpc_id
  subnets = module.vpc.public_subnets

  security_group_ingress_rules = {
    all_http = {
      from_port   = 80
      to_port     = 80
      ip_protocol = "tcp"
      cidr_ipv4   = "0.0.0.0/0"
    }
    all_https = {
      from_port   = 443
      to_port     = 443
      ip_protocol = "tcp"
      cidr_ipv4   = "0.0.0.0/0"
    }
  }

  tags = {
    Environment = "production"
    Project     = "reasonkit"
  }
}
```

### GCP Deployment

#### Cloud Run

```bash
# Build and push to Artifact Registry
gcloud auth configure-docker us-central1-docker.pkg.dev

docker tag reasonkit/core:latest \
  us-central1-docker.pkg.dev/PROJECT_ID/reasonkit/core:latest

docker push us-central1-docker.pkg.dev/PROJECT_ID/reasonkit/core:latest

# Deploy to Cloud Run
gcloud run deploy reasonkit-core \
  --image us-central1-docker.pkg.dev/PROJECT_ID/reasonkit/core:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \
  --max-instances 10 \
  --port 8080 \
  --set-env-vars "RUST_LOG=info,REASONKIT_ENV=production" \
  --set-secrets "ANTHROPIC_API_KEY=reasonkit-anthropic-key:latest"
```

#### GKE (Google Kubernetes Engine)

```bash
# Create GKE cluster
gcloud container clusters create reasonkit-cluster \
  --zone us-central1-a \
  --num-nodes 3 \
  --machine-type e2-standard-4 \
  --enable-autoscaling \
  --min-nodes 2 \
  --max-nodes 10 \
  --enable-autorepair \
  --enable-autoupgrade

# Get credentials
gcloud container clusters get-credentials reasonkit-cluster --zone us-central1-a

# Apply Kubernetes manifests
kubectl apply -k ./k8s/overlays/gcp/
```

#### Terraform (GCP)

```hcl
# main.tf
terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
  }
}

provider "google" {
  project = var.project_id
  region  = var.region
}

# VPC Network
resource "google_compute_network" "reasonkit" {
  name                    = "reasonkit-network"
  auto_create_subnetworks = false
}

resource "google_compute_subnetwork" "reasonkit" {
  name          = "reasonkit-subnet"
  ip_cidr_range = "10.0.0.0/24"
  region        = var.region
  network       = google_compute_network.reasonkit.id

  secondary_ip_range {
    range_name    = "pods"
    ip_cidr_range = "10.1.0.0/16"
  }

  secondary_ip_range {
    range_name    = "services"
    ip_cidr_range = "10.2.0.0/20"
  }
}

# GKE Cluster
resource "google_container_cluster" "reasonkit" {
  name     = "reasonkit-cluster"
  location = var.zone

  network    = google_compute_network.reasonkit.name
  subnetwork = google_compute_subnetwork.reasonkit.name

  remove_default_node_pool = true
  initial_node_count       = 1

  ip_allocation_policy {
    cluster_secondary_range_name  = "pods"
    services_secondary_range_name = "services"
  }

  workload_identity_config {
    workload_pool = "${var.project_id}.svc.id.goog"
  }
}

resource "google_container_node_pool" "reasonkit" {
  name       = "reasonkit-pool"
  location   = var.zone
  cluster    = google_container_cluster.reasonkit.name
  node_count = 3

  node_config {
    machine_type = "e2-standard-4"
    disk_size_gb = 100
    disk_type    = "pd-ssd"

    oauth_scopes = [
      "https://www.googleapis.com/auth/cloud-platform"
    ]
  }

  autoscaling {
    min_node_count = 2
    max_node_count = 10
  }
}

# Secret Manager
resource "google_secret_manager_secret" "anthropic_key" {
  secret_id = "reasonkit-anthropic-key"

  replication {
    auto {}
  }
}
```

### Azure Deployment

#### Azure Container Instances

```bash
# Create resource group
az group create --name reasonkit-rg --location eastus

# Create container instance
az container create \
  --resource-group reasonkit-rg \
  --name reasonkit-core \
  --image reasonkit/core:latest \
  --cpu 2 \
  --memory 4 \
  --ports 8080 \
  --dns-name-label reasonkit-api \
  --environment-variables \
    RUST_LOG=info \
    REASONKIT_ENV=production \
  --secure-environment-variables \
    ANTHROPIC_API_KEY=$ANTHROPIC_API_KEY
```

#### AKS (Azure Kubernetes Service)

```bash
# Create AKS cluster
az aks create \
  --resource-group reasonkit-rg \
  --name reasonkit-cluster \
  --node-count 3 \
  --node-vm-size Standard_D4s_v3 \
  --enable-cluster-autoscaler \
  --min-count 2 \
  --max-count 10 \
  --enable-managed-identity \
  --enable-addons monitoring

# Get credentials
az aks get-credentials --resource-group reasonkit-rg --name reasonkit-cluster

# Apply Kubernetes manifests
kubectl apply -k ./k8s/overlays/azure/
```

#### Terraform (Azure)

```hcl
# main.tf
terraform {
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "reasonkit" {
  name     = "reasonkit-rg"
  location = var.location
}

resource "azurerm_kubernetes_cluster" "reasonkit" {
  name                = "reasonkit-cluster"
  location            = azurerm_resource_group.reasonkit.location
  resource_group_name = azurerm_resource_group.reasonkit.name
  dns_prefix          = "reasonkit"

  default_node_pool {
    name                = "default"
    node_count          = 3
    vm_size             = "Standard_D4s_v3"
    enable_auto_scaling = true
    min_count           = 2
    max_count           = 10
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }

  tags = {
    Environment = "Production"
    Project     = "ReasonKit"
  }
}

resource "azurerm_key_vault" "reasonkit" {
  name                = "reasonkit-vault"
  location            = azurerm_resource_group.reasonkit.location
  resource_group_name = azurerm_resource_group.reasonkit.name
  tenant_id           = data.azurerm_client_config.current.tenant_id
  sku_name            = "standard"
}
```

---

## Environment Configuration

### Configuration Hierarchy

ReasonKit uses a layered configuration system:

1. **Defaults** - Built-in sensible defaults
2. **Config File** - `/app/config/default.toml`
3. **Environment Variables** - Override any setting
4. **CLI Arguments** - Highest priority

### Core Environment Variables

| Variable         | Description                                    | Default       | Required |
| ---------------- | ---------------------------------------------- | ------------- | -------- |
| `RUST_LOG`       | Log level (trace, debug, info, warn, error)    | `info`        | No       |
| `REASONKIT_ENV`  | Environment (development, staging, production) | `development` | No       |
| `SERVER_HOST`    | Bind address                                   | `127.0.0.1`   | No       |
| `SERVER_PORT`    | Listen port                                    | `8080`        | No       |
| `SERVER_WORKERS` | Worker threads                                 | CPU count     | No       |

### LLM Provider Configuration

| Variable               | Description                                      | Required     |
| ---------------------- | ------------------------------------------------ | ------------ |
| `ANTHROPIC_API_KEY`    | Anthropic Claude API key                         | At least one |
| `OPENAI_API_KEY`       | OpenAI API key                                   | At least one |
| `OPENROUTER_API_KEY`   | OpenRouter API key                               | At least one |
| `LLM_DEFAULT_PROVIDER` | Default provider (anthropic, openai, openrouter) | No           |
| `LLM_DEFAULT_MODEL`    | Default model ID                                 | No           |
| `LLM_TEMPERATURE`      | Default temperature (0.0-2.0)                    | No           |
| `LLM_MAX_TOKENS`       | Default max tokens                               | No           |

### Database Configuration

| Variable         | Description          | Default                  |
| ---------------- | -------------------- | ------------------------ |
| `QDRANT_URL`     | Qdrant vector DB URL | `http://localhost:6333`  |
| `QDRANT_API_KEY` | Qdrant API key       | None                     |
| `REDIS_URL`      | Redis cache URL      | `redis://localhost:6379` |

### Feature Flags

| Variable               | Description                | Default |
| ---------------------- | -------------------------- | ------- |
| `ENABLE_METRICS`       | Enable Prometheus metrics  | `false` |
| `ENABLE_TRACING`       | Enable distributed tracing | `false` |
| `ENABLE_HEALTH_CHECKS` | Enable health endpoints    | `true`  |
| `ENABLE_CACHE`         | Enable response caching    | `true`  |

### Configuration File Example

```toml
# /app/config/default.toml

[server]
host = "0.0.0.0"
port = 8080
workers = 4
timeout_secs = 300

[logging]
level = "info"
format = "json"
include_timestamps = true

[llm]
default_provider = "anthropic"
default_model = "claude-sonnet-4-20250514"
temperature = 0.7
max_tokens = 2000

[llm.providers.anthropic]
api_base = "https://api.anthropic.com"
timeout_secs = 120

[llm.providers.openai]
api_base = "https://api.openai.com/v1"
timeout_secs = 120

[database.qdrant]
url = "http://localhost:6333"
collection_prefix = "reasonkit_"

[database.redis]
url = "redis://localhost:6379"
max_connections = 10

[features]
enable_metrics = true
enable_tracing = true
enable_cache = true

[security]
allowed_origins = ["https://reasonkit.sh", "https://api.reasonkit.sh"]
rate_limit_per_minute = 100
max_request_size_mb = 50
```

---

## Health Checks

### Endpoints

| Endpoint        | Method | Description        | Response                              |
| --------------- | ------ | ------------------ | ------------------------------------- |
| `/health`       | GET    | Basic health check | `200 OK` or `503 Service Unavailable` |
| `/health/live`  | GET    | Liveness probe     | `200 OK` if process is running        |
| `/health/ready` | GET    | Readiness probe    | `200 OK` if ready to accept traffic   |
| `/metrics`      | GET    | Prometheus metrics | Prometheus format                     |

### Health Response Format

```json
{
  "status": "healthy",
  "version": "0.1.0",
  "timestamp": "2026-01-01T00:00:00Z",
  "checks": {
    "llm_provider": {
      "status": "healthy",
      "latency_ms": 150
    },
    "qdrant": {
      "status": "healthy",
      "latency_ms": 5
    },
    "redis": {
      "status": "healthy",
      "latency_ms": 1
    }
  }
}
```

### Docker Health Check

```dockerfile
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD rk --version || exit 1
```

### Kubernetes Probes

```yaml
livenessProbe:
  exec:
    command:
      - rk
      - --version
  initialDelaySeconds: 10
  periodSeconds: 30
  timeoutSeconds: 5
  failureThreshold: 3

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
  timeoutSeconds: 3
  failureThreshold: 3

startupProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 0
  periodSeconds: 5
  timeoutSeconds: 3
  failureThreshold: 30
```

---

## Scaling Considerations

### Horizontal Scaling

ReasonKit is designed for horizontal scaling. Each instance is stateless.

**Scaling Triggers:**

- CPU utilization > 70%
- Memory utilization > 80%
- Request latency p99 > 500ms
- Request queue depth > 100

**Kubernetes HPA Example:**

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: reasonkit-core-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: reasonkit-core
  minReplicas: 3
  maxReplicas: 50
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 80
    - type: Pods
      pods:
        metric:
          name: http_requests_per_second
        target:
          type: AverageValue
          averageValue: 100
```

### Vertical Scaling

For single-instance deployments:

| Workload                | CPU      | Memory | Storage |
| ----------------------- | -------- | ------ | ------- |
| Development             | 2 cores  | 2 GB   | 10 GB   |
| Small (< 100 req/min)   | 4 cores  | 4 GB   | 50 GB   |
| Medium (< 1000 req/min) | 8 cores  | 8 GB   | 100 GB  |
| Large (< 10000 req/min) | 16 cores | 16 GB  | 500 GB  |

### Database Scaling

**Qdrant:**

- Horizontal: Cluster mode with sharding
- Vertical: Increase memory for larger indices

**Redis:**

- Horizontal: Redis Cluster for > 100GB data
- Vertical: Increase memory for hot cache

### LLM Provider Rate Limits

Design for provider rate limits:

| Provider   | Default Limit | Strategy      |
| ---------- | ------------- | ------------- |
| Anthropic  | 60 req/min    | Queue + retry |
| OpenAI     | 500 req/min   | Round-robin   |
| OpenRouter | 200 req/min   | Fallback      |

**Rate Limit Handling:**

```yaml
[llm.rate_limiting]
strategy = "queue"  # queue, drop, fallback
queue_size = 1000
retry_attempts = 3
retry_delay_ms = 1000
fallback_provider = "openrouter"
```

---

## Security Hardening

### Container Security

```dockerfile
# Run as non-root user
USER 1000:1000

# Read-only filesystem
# Use tmpfs for /tmp
```

```yaml
# Kubernetes SecurityContext
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  runAsGroup: 1000
  fsGroup: 1000
  allowPrivilegeEscalation: false
  readOnlyRootFilesystem: true
  seccompProfile:
    type: RuntimeDefault
  capabilities:
    drop:
      - ALL
```

### Network Security

```yaml
# NetworkPolicy - Allow only necessary traffic
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: reasonkit-network-policy
  namespace: reasonkit
spec:
  podSelector:
    matchLabels:
      app: reasonkit-core
  policyTypes:
    - Ingress
    - Egress
  ingress:
    - from:
        - namespaceSelector:
            matchLabels:
              name: ingress-nginx
      ports:
        - protocol: TCP
          port: 8080
  egress:
    # Allow DNS
    - to:
        - namespaceSelector: {}
      ports:
        - protocol: UDP
          port: 53
    # Allow LLM providers
    - to:
        - ipBlock:
            cidr: 0.0.0.0/0
      ports:
        - protocol: TCP
          port: 443
    # Allow internal services
    - to:
        - podSelector:
            matchLabels:
              app: qdrant
      ports:
        - protocol: TCP
          port: 6333
    - to:
        - podSelector:
            matchLabels:
              app: redis
      ports:
        - protocol: TCP
          port: 6379
```

### Secret Management

**Never store secrets in:**

- Docker images
- ConfigMaps
- Environment variables in manifests
- Version control

**Use:**

- Kubernetes Secrets (encrypted at rest)
- HashiCorp Vault
- AWS Secrets Manager
- GCP Secret Manager
- Azure Key Vault

**External Secrets Operator Example:**

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: reasonkit-secrets
  namespace: reasonkit
spec:
  refreshInterval: 1h
  secretStoreRef:
    name: vault-backend
    kind: SecretStore
  target:
    name: reasonkit-secrets
    creationPolicy: Owner
  data:
    - secretKey: ANTHROPIC_API_KEY
      remoteRef:
        key: secret/data/reasonkit
        property: anthropic_api_key
```

### TLS/SSL Configuration

```yaml
# Ingress with TLS
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: reasonkit-ingress
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/force-ssl-redirect: "true"
spec:
  tls:
    - hosts:
        - api.reasonkit.sh
      secretName: reasonkit-tls
  rules:
    - host: api.reasonkit.sh
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: reasonkit-core
                port:
                  number: 8080
```

---

## Monitoring & Observability

### Prometheus Metrics

ReasonKit exposes metrics at `/metrics`:

```
# HELP reasonkit_requests_total Total number of requests
# TYPE reasonkit_requests_total counter
reasonkit_requests_total{method="think",profile="balanced"} 1523

# HELP reasonkit_request_duration_seconds Request latency histogram
# TYPE reasonkit_request_duration_seconds histogram
reasonkit_request_duration_seconds_bucket{le="0.1"} 50
reasonkit_request_duration_seconds_bucket{le="0.5"} 200
reasonkit_request_duration_seconds_bucket{le="1.0"} 500

# HELP reasonkit_llm_tokens_total Total LLM tokens used
# TYPE reasonkit_llm_tokens_total counter
reasonkit_llm_tokens_total{provider="anthropic",type="input"} 150000
reasonkit_llm_tokens_total{provider="anthropic",type="output"} 75000

# HELP reasonkit_confidence_score Confidence score distribution
# TYPE reasonkit_confidence_score histogram
reasonkit_confidence_score_bucket{profile="balanced",le="0.7"} 10
reasonkit_confidence_score_bucket{profile="balanced",le="0.8"} 50
reasonkit_confidence_score_bucket{profile="balanced",le="0.9"} 150
```

### Prometheus Configuration

```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "reasonkit-core"
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_label_app]
        action: keep
        regex: reasonkit-core
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_port]
        action: replace
        target_label: __address__
        regex: (.+)
        replacement: $1
```

### Grafana Dashboard

Import the ReasonKit dashboard from `monitoring/grafana/dashboards/reasonkit-overview.json` or use dashboard ID `XXXXX` from Grafana.com.

**Key Panels:**

- Request rate and latency (p50, p95, p99)
- Error rate by type
- LLM token usage and costs
- Confidence score distribution
- ThinkTool execution times
- Cache hit/miss ratio

### Alerting Rules

```yaml
# alerts.yml
groups:
  - name: reasonkit
    rules:
      - alert: ReasonKitHighErrorRate
        expr: |
          sum(rate(reasonkit_requests_total{status="error"}[5m]))
          /
          sum(rate(reasonkit_requests_total[5m])) > 0.05
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is above 5% for 5 minutes"

      - alert: ReasonKitHighLatency
        expr: |
          histogram_quantile(0.99, sum(rate(reasonkit_request_duration_seconds_bucket[5m])) by (le)) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High latency detected"
          description: "P99 latency is above 5 seconds"

      - alert: ReasonKitLowConfidence
        expr: |
          histogram_quantile(0.5, sum(rate(reasonkit_confidence_score_bucket[1h])) by (le)) < 0.7
        for: 1h
        labels:
          severity: warning
        annotations:
          summary: "Low confidence scores"
          description: "Median confidence score is below 70%"
```

### Structured Logging

```json
{
  "timestamp": "2026-01-01T00:00:00.000Z",
  "level": "info",
  "target": "reasonkit::thinktool::executor",
  "message": "Protocol execution completed",
  "span": {
    "request_id": "req_abc123",
    "profile": "balanced",
    "duration_ms": 2345
  },
  "fields": {
    "confidence": 0.87,
    "tools_executed": ["GigaThink", "LaserLogic", "ProofGuard"],
    "llm_provider": "anthropic",
    "tokens_used": 1500
  }
}
```

---

## Troubleshooting

### Common Issues

#### Container Won't Start

```bash
# Check logs
docker logs reasonkit-core

# Common causes:
# 1. Missing API keys
# 2. Port already in use
# 3. Insufficient memory
```

#### High Memory Usage

```bash
# Check current usage
docker stats reasonkit-core

# Solutions:
# 1. Increase memory limits
# 2. Enable swap (not recommended for production)
# 3. Reduce worker count
# 4. Enable response streaming
```

#### LLM Provider Errors

```bash
# Check provider connectivity
curl -I https://api.anthropic.com/v1/messages

# Common causes:
# 1. Invalid API key
# 2. Rate limiting
# 3. Network issues
# 4. Provider outage
```

#### Slow Response Times

```bash
# Check metrics
curl localhost:9100/metrics | grep duration

# Common causes:
# 1. Insufficient CPU
# 2. Cold starts (increase min replicas)
# 3. LLM provider latency
# 4. Large input payloads
```

### Debug Mode

```bash
# Enable debug logging
docker run -e RUST_LOG=debug reasonkit/core:latest

# Enable trace logging (very verbose)
docker run -e RUST_LOG=trace reasonkit/core:latest

# Log specific modules
docker run -e RUST_LOG=reasonkit=debug,reqwest=info reasonkit/core:latest
```

### Performance Profiling

```bash
# CPU profiling with perf
docker run --privileged --cap-add SYS_ADMIN reasonkit/core:latest

# Memory profiling with heaptrack
docker run -v /tmp/heap:/tmp/heap reasonkit/core:latest \
  heaptrack rk serve-mcp
```

### Getting Help

1. Check the [documentation](https://docs.rs/reasonkit-core)
2. Search [GitHub Issues](https://github.com/reasonkit/reasonkit-core/issues)
3. Ask in [GitHub Discussions](https://github.com/reasonkit/reasonkit-core/discussions)
4. Email support: support@reasonkit.sh

---

## Version History

| Version | Date       | Changes         |
| ------- | ---------- | --------------- |
| 1.0.0   | 2026-01-01 | Initial release |

---

_ReasonKit - Turn Prompts into Protocols_
*https://reasonkit.sh*
