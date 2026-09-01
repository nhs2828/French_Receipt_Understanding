# Minikube demo deploy
## 0. Start minikube with enough resources
```bash
# 24576
# cpu
minikube start --driver=docker --cpus=4 --memory=8192
# gpu
minikube start --driver=docker --cpus=4 --memory=8192 --gpus=all
minikube addons enable nvidia-device-plugin
```
Increase `--memory` if the vision-service pod OOMKills — PaddleOCR + YOLO-seg are heavy.

<!-- ## Optional
## 1. Enable the in-cluster registry
```bash
minikube addons enable registry
```

In a **separate terminal, left running**, (macOS/docker driver needs
this because the VM's network isn't directly reachable from the host):
```bash
docker run --rm -it --network=host alpine ash -c \
  "apk add socat && socat TCP-LISTEN:5000,reuseaddr,fork TCP:$(minikube ip):5000"
``` -->

## 1. Build and push both images
## Replace with Dockefile_cpu if want to run with cpu
From the repo root:
```bash
docker build -t nhs2828/vision-service-cpu:v1.0 \
  -f services/vision-service/Dockerfile_cpu services/vision-service
docker push nhs2828/vision-service-cpu:latest

docker build -t nhs2828/kie-service-cpu:v1.0 \
  -f services/kie-service/Dockerfile_cpu .
docker push custom_name_here/kie-service:latest
```
Re-run these two `build` + `push` pairs any time we change code

## 2. Mount local model weights into minikube
### Local model test
Two separate terminals, **left running** (each is a foreground process):
```bash
minikube mount ./services/vision-service/models:/mnt/host-models/vision
minikube mount ./services/kie-service/models:/mnt/host-models/kie
```

### Cloud - Create the S3 credentials secret and configure your bucket

```bash
kubectl create namespace receipt-understanding --dry-run=client -o yaml | kubectl apply -f -
kubectl create secret generic s3-model-creds \
  --namespace receipt-understanding \
  --from-literal=AWS_ACCESS_KEY_ID=<your-key-id> \
  --from-literal=AWS_SECRET_ACCESS_KEY=<your-secret-key>
```
Then edit k8s/vision-service/configmap.yaml and k8s/kie-service/configmap.yaml
 — replace S3_BUCKET, S3_MODEL_PREFIX, and AWS_REGION with your real bucket details. 
 Each Deployment's pull-models init container downloads the weights fresh into 
 an emptyDir on every pod start — no more minikube mount, no host filesystem dependency.

Startup will take longer now — model download time gets added on top of model load time. 
The readiness probes already account for this with generous initialDelaySeconds/failureThreshold, 
but watch kubectl logs -n receipt-understanding <pod> -c pull-models if a pod seems stuck.

## 3. Deploy the services
Skip this if use Helm to deploy
### local test
```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/vision-service/configmap.yaml
kubectl apply -f k8s/vision-service/deployment.yaml
kubectl apply -f k8s/vision-service/service.yaml
kubectl apply -f k8s/kie-service/configmap.yaml
kubectl apply -f k8s/kie-service/deployment.yaml
kubectl apply -f k8s/kie-service/service.yaml
```
### Cloud aws
```bash
kubectl apply -f k8s/00-namespace.yaml
kubectl apply -f k8s/vision-service/configmap_cloud.yaml
kubectl apply -f k8s/vision-service/deployment_cloud.yaml
kubectl apply -f k8s/vision-service/service.yaml
kubectl apply -f k8s/kie-service/configmap_cloud.yaml
kubectl apply -f k8s/kie-service/deployment_cloud.yaml
kubectl apply -f k8s/kie-service/service.yaml
```

## 4. Install the monitoring stack
```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo add grafana https://grafana.github.io/helm-charts
helm repo update

helm install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  -f k8s/monitoring/kube-prometheus-stack-values.yaml

helm install loki grafana/loki-stack \
  -n monitoring \
  -f k8s/monitoring/loki-stack-values.yaml
```
The release name **must** be `monitoring` — the ServiceMonitors' `release: monitoring` label
depends on it for auto-discovery.

## 5. Load the dashboard and wire up scraping
### no Helm
```bash
kubectl apply -f k8s/monitoring/grafana-dashboard-configmap.yaml
kubectl apply -f k8s/vision-service/servicemonitor.yaml
kubectl apply -f k8s/kie-service/servicemonitor.yaml
```
### With Helm
```bash
helm upgrade --install receipt-understanding k8s/helm/receipt-understanding \
  -n receipt-understanding --create-namespace \
  -f k8s/helm/receipt-understanding/values.yaml \
  -f k8s/helm/receipt-understanding/values-cloud-gpu.yaml
```

Watch startup:
```bash
kubectl get pods -n receipt-understanding -w
```

## 6. Access everything
## add more port forward to observer other services
```bash
# Get the endpoints url
minikube service kie-service -n receipt-understanding --url
# in nested VM
kubectl port-forward svc/kie-service 8000:8000 -n receipt-understanding

# Grafana (admin / admin, or whatever set in the values file)
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Prometheus, for raw queries
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```
## 7.Shutdown
```bash
minikube stop
minikube delete --all
```

## Quick troubleshooting
- `kubectl get pods -n receipt-understanding` stuck in `ImagePullBackOff` → the registry
  push/socat forward from step 1 likely isn't running, or the image wasn't actually pushed.
- Pod `CrashLoopBackOff` right after start → check `kubectl logs -n receipt-understanding
  <pod>`
- kubectl logs <pod> -n receipt-understanding to check pod logs
- kubectl describe <pod> -n receipt-understanding to check startup

## Misc setup fresh machine (linux/ubuntu)
For people like me who couldn't remember all the commands
### AWS CLI
```bash
curl -fsSL https://awscli.amazonaws.com/v2/install.sh | bash
export PATH=$PATH:/root/.local/bin
```

### Install minikube
```bash
curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-linux-amd64
sudo install minikube-linux-amd64 /usr/local/bin/minikube
rm minikube-linux-amd64
```

### Install NVIDIA toolkit for nvidia gpu
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \ sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \ tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt update
apt install -y nvidia-container-toolkit
```

### kubectl
```bash
curl -LO "https://dl.k8s.io/release/$(curl -L -s https://dl.k8s.io/release/stable.txt)/bin/linux/amd64/kubectl"
sudo install -o root -g root -m 0755 kubectl /usr/local/bin/kubectl
rm kubectl
```

### Helm
```bash
curl https://raw.githubusercontent.com/helm/helm/main/scripts/get-helm-3 | bash
```
