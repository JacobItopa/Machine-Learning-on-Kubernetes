import openshift as oc
import os, time
from jinja2 import Template
from mlflow.tracking import MlflowClient
import mlflow

HOST = "http://mlflow:5500"

model_name = os.environ["MODEL_NAME"]
model_version = os.environ["MODEL_VERSION"]
cluster_dns = os.environ["CLUSTER_DOMAIN_NAME"]

ingress_host = f"{model_name}.{cluster_dns}"

build_name = f"seldon-model-{model_name}-v{model_version}"

mlflow.set_tracking_uri(HOST)

def get_model_run_info(host, model_name, model_version):
    import json
    import urllib.request
    import os

    if os.environ.get("RUN_ID") and os.environ.get("EXPERIMENT_ID"):
        return os.environ["RUN_ID"], os.environ["EXPERIMENT_ID"]

    def _fetch_json(url, post_data=None):
        try:
            req = urllib.request.Request(url, headers={"Content-Type": "application/json"})
            body = json.dumps(post_data).encode("utf-8") if post_data is not None else None
            with urllib.request.urlopen(req, data=body, timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as e:
            return None

    # 1. Try direct REST endpoints
    endpoints = [
        f"{host}/api/2.0/mlflow/model-versions/get?name={model_name}&version={model_version}",
        f"{host}/api/2.0/mlflow/registered-models/get?name={model_name}",
    ]
    for ep in endpoints:
        data = _fetch_json(ep)
        if data:
            if "model_version" in data:
                mv = data["model_version"]
                exp_id = mv.get("source", "").split("/")[3] if len(mv.get("source", "").split("/")) > 3 else "1"
                print(f"Matched model version: run_id={mv.get('run_id')}, exp_id={exp_id}")
                return mv.get("run_id"), exp_id
            elif "registered_model" in data:
                rm = data["registered_model"]
                for lv in rm.get("latest_versions", []):
                    if str(lv.get("version")) == str(model_version):
                        exp_id = lv.get("source", "").split("/")[3] if len(lv.get("source", "").split("/")) > 3 else "1"
                        print(f"Matched registered model: run_id={lv.get('run_id')}, exp_id={exp_id}")
                        return lv.get("run_id"), exp_id

    # 2. Search all registered models (case-insensitive & fuzzy)
    print("Direct model lookup returned 404, searching all registered models in MLflow...")
    search_data = _fetch_json(f"{host}/api/2.0/mlflow/registered-models/search")
    if search_data and "registered_models" in search_data:
        models = search_data["registered_models"]
        names = [m.get("name") for m in models]
        print(f"Registered models found in MLflow: {names}")

        target = model_name.lower().replace("-", "").replace("_", "")
        for m in models:
            m_norm = m.get("name", "").lower().replace("-", "").replace("_", "")
            if m_norm == target or target in m_norm or m_norm in target:
                for lv in m.get("latest_versions", []):
                    exp_id = lv.get("source", "").split("/")[3] if len(lv.get("source", "").split("/")) > 3 else "1"
                    print(f"Fuzzy matched model '{m.get('name')}': run_id={lv.get('run_id')}, exp_id={exp_id}")
                    return lv.get("run_id"), exp_id

        if len(models) == 1:
            m = models[0]
            lv = m.get("latest_versions", [{}])[0]
            exp_id = lv.get("source", "").split("/")[3] if len(lv.get("source", "").split("/")) > 3 else "1"
            print(f"Using the single registered model in MLflow '{m.get('name')}': run_id={lv.get('run_id')}, exp_id={exp_id}")
            return lv.get("run_id"), exp_id

    # 3. Fallback: Search experiment runs
    print("Searching active experiment runs in MLflow for logged artifacts...")
    exp_data = _fetch_json(f"{host}/api/2.0/mlflow/experiments/search")
    exp_ids = []
    if exp_data and "experiments" in exp_data:
        exp_ids = [e.get("experiment_id") for e in exp_data["experiments"] if e.get("lifecycle_stage") == "active"]
    if not exp_ids:
        exp_ids = ["0", "1", "2"]

    runs_data = _fetch_json(f"{host}/api/2.0/mlflow/runs/search", post_data={"experiment_ids": exp_ids, "max_results": 10})
    if runs_data and "runs" in runs_data and len(runs_data["runs"]) > 0:
        latest_run = runs_data["runs"][0]
        run_info = latest_run.get("info", {})
        run_id = run_info.get("run_id")
        exp_id = run_info.get("experiment_id", "1")
        print(f"Auto-selected latest experiment run: run_id={run_id}, exp_id={exp_id}")
        return run_id, exp_id

    raise RuntimeError(f"Could not find any model or active run in MLflow tracking server {host}")

run_id, experiment_id = get_model_run_info(HOST, model_name, model_version)
print(f"Resolved metadata: run_id={run_id}, experiment_id={experiment_id}")


model_container_location = os.environ["CONTAINER_DETAILS"]

if os.environ.get("CONTAINER_REGISTRY") is not None and os.environ["CONTAINER_REGISTRY"].find("docker.io") == -1:
    model_container_location = os.environ["CONTAINER_REGISTRY"] + "/" + os.environ["CONTAINER_DETAILS"]

server = "https://" + os.environ["KUBERNETES_SERVICE_HOST"] + ":" + os.environ["KUBERNETES_SERVICE_PORT"]
print(server)

#build from source Docker file
with open('/var/run/secrets/kubernetes.io/serviceaccount/token', 'r') as file:
    token = file.read()
print(f"Access Token{token}")

#/var/run/secrets/kubernetes.io/serviceaccount
with open('/var/run/secrets/kubernetes.io/serviceaccount/namespace', 'r') as namespace:
    project = namespace.read()
print(f"Namespace: {project}")

with oc.api_server(server):
    with oc.token(token):
        with oc.project(project), oc.timeout(10*60):
            template_data = {"experiment_id": run_id, "model_name": model_name, "model_coordinates": model_container_location, "ingress_host": ingress_host}
            applied_template = Template(open("SeldonDeploy.yaml").read())
            
            rendered_template = applied_template.render(template_data)
            print('Rendered Template: \r' + rendered_template)
            oc.apply(rendered_template)
            
            service_name = "model-" + run_id + "-" + model_name
            ingress_count = oc.selector(f"ingress/{service_name}").count_existing()
            print(ingress_count)
            if ingress_count == 0:
                service_name = "model-" + run_id + "-" + model_name
                while True:
                    service_count = oc.selector(f"service/{service_name}").count_existing()
                    if service_count > 0:
                        service = oc.selector(f"service/{service_name}").object()
                        print(service.name())
                        applied_template = Template(open("Ingress.yaml").read())
                        print(applied_template.render(template_data))
                        oc.apply(applied_template.render(template_data))
                        break
                    else:
                        print(f"Service name does not exist {service_name}")
                        time.sleep(10)
            else:
                print(f"Ingress already exists for {service_name}")