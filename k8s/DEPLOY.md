# Minikube demo deploy
## 0. Start minikube with enough resources
```bash
minikube start --driver=docker --cpus=4 --memory=8192
```
Increase `--memory` if the vision-service pod OOMKills — PaddleOCR + YOLO-seg are heavy.

## Optional
## 1. Enable the in-cluster registry
```bash
minikube addons enable registry
```

In a **separate terminal, left running**, (macOS/docker driver needs
this because the VM's network isn't directly reachable from the host):
```bash
docker run --rm -it --network=host alpine ash -c \
  "apk add socat && socat TCP-LISTEN:5000,reuseaddr,fork TCP:$(minikube ip):5000"
```

## 2. Build and push both images
## Replace with Dockefile_cpu if want to run with cpu
From the repo root:
```bash
docker build -t custom_name_here/vision-service:latest \
  -f services/vision-service/Dockerfile services/vision-service
docker push custom_name_here/vision-service:latest

docker build -t custom_name_here/kie-service:latest \
  -f services/kie-service/Dockerfile .
docker push custom_name_here/kie-service:latest
```
Re-run these two `build` + `push` pairs any time we change code

## 3. Mount local model weights into minikube
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

## 4. Deploy the services
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

## 5. Install the monitoring stack
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

## 6. Load the dashboard and wire up scraping
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
  -f values-cloud-gpu.yaml
```

Watch startup:
```bash
kubectl get pods -n receipt-understanding -w
```

## 7. Access everything
## add more port forward to observer other services
```bash
# Get the endpoints url
minikube service kie-service -n receipt-understanding --url

# Grafana (admin / admin, or whatever set in the values file)
kubectl port-forward -n monitoring svc/monitoring-grafana 3000:80

# Prometheus, for raw queries
kubectl port-forward -n monitoring svc/monitoring-kube-prometheus-prometheus 9090:9090
```

## Quick troubleshooting
- `kubectl get pods -n receipt-understanding` stuck in `ImagePullBackOff` → the registry
  push/socat forward from step 1 likely isn't running, or the image wasn't actually pushed.
- Pod `CrashLoopBackOff` right after start → check `kubectl logs -n receipt-understanding
  <pod>`
- kubectl logs <pod> -n receipt-understanding to check pod logs
