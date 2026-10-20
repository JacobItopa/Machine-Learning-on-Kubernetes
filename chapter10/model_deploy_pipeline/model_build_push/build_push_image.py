import string
import subprocess
import os
import base64
import mlflow
from minio import Minio
from mlflow.tracking import MlflowClient


"""
    This script assumes that the /kaniko/.docker/config.json has the correct repo and associated credentials mounted
    It also expects the these env variables has been set
    CONTAINER_REGISTRY is the resitry server like quay.io
    CONTAINER_DETAILS is the container coordinates like ml-on-k8s/containermodel:1.0.0
    AWS_SECRET_ACCESS_KEY is the password for the S3 store
    MODEL_NAME is hte name of the model in mlflow
    MODEL_VERSION is the version of the model in mlflow
"""

os.environ['MLFLOW_S3_ENDPOINT_URL']='http://minio-ml-workshop:9000'
os.environ['AWS_ACCESS_KEY_ID']='minio'
os.environ['AWS_REGION']='us-east-1'
os.environ['AWS_BUCKET_NAME']='mlflow'

HOST = "http://mlflow:5500"

model_name = os.environ["MODEL_NAME"]
model_version = os.environ["MODEL_VERSION"]
build_name = f"seldon-model-{model_name}-v{model_version}"

auth_encoded = f"{os.environ['CONTAINER_REGISTRY_USER']}:{os.environ['CONTAINER_REGISTRY_PASSWORD']}"
creds = base64.b64encode(auth_encoded.encode("ascii")).decode("ascii")

reg = os.environ.get("CONTAINER_REGISTRY", "https://index.docker.io/v1/")
auth_dict = {
    reg: {"auth": creds},
    "https://index.docker.io/v1/": {"auth": creds},
    "index.docker.io": {"auth": creds},
    "docker.io": {"auth": creds},
}
docker_auth = json.dumps({"auths": auth_dict})
os.makedirs("/kaniko/.docker", exist_ok=True)
with open("/kaniko/.docker/config.json", "w") as f:
    f.write(docker_auth)

def get_s3_server():
    minioClient = Minio('minio-ml-workshop:9000',
                        access_key=os.environ['AWS_ACCESS_KEY_ID'],
                        secret_key=os.environ["AWS_SECRET_ACCESS_KEY"],
                        secure=False)

    return minioClient


def init():
    mlflow.set_tracking_uri(HOST)


def get_model_run_info(host, model_name, model_version):
    import json
    import urllib.request
    import os

    if os.environ.get("RUN_ID") and os.environ.get("EXPERIMENT_ID"):
        return os.environ["RUN_ID"], os.environ["EXPERIMENT_ID"], f"s3://mlflow/{os.environ['EXPERIMENT_ID']}/{os.environ['RUN_ID']}/artifacts/model"

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
                return mv.get("run_id"), exp_id, mv.get("source", "")
            elif "registered_model" in data:
                rm = data["registered_model"]
                for lv in rm.get("latest_versions", []):
                    if str(lv.get("version")) == str(model_version):
                        exp_id = lv.get("source", "").split("/")[3] if len(lv.get("source", "").split("/")) > 3 else "1"
                        print(f"Matched registered model: run_id={lv.get('run_id')}, exp_id={exp_id}")
                        return lv.get("run_id"), exp_id, lv.get("source", "")

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
                    return lv.get("run_id"), exp_id, lv.get("source", "")

        if len(models) == 1:
            m = models[0]
            lv = m.get("latest_versions", [{}])[0]
            exp_id = lv.get("source", "").split("/")[3] if len(lv.get("source", "").split("/")) > 3 else "1"
            print(f"Using the single registered model in MLflow '{m.get('name')}': run_id={lv.get('run_id')}, exp_id={exp_id}")
            return lv.get("run_id"), exp_id, lv.get("source", "")

    # 3. Fallback: Search experiment runs (if model was logged but not registered under Models tab)
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
        artifact_uri = run_info.get("artifact_uri", f"s3://mlflow/{exp_id}/{run_id}/artifacts")
        source = f"{artifact_uri}/model"
        print(f"Auto-selected latest experiment run: run_id={run_id}, exp_id={exp_id}, source={source}")
        return run_id, exp_id, source

    raise RuntimeError(f"Could not find any model or active run in MLflow tracking server {host}")


def download_artifacts():
    print("retrieving model metadata from mlflow...")
    run_id, experiment_id, source = get_model_run_info(HOST, model_name, model_version)
    print(f"Found model metadata: run_id={run_id}, experiment_id={experiment_id}, source={source}")

    print("initializing connection to s3 server...")
    minioClient = get_s3_server()

    #     artifact_location = mlflow.get_experiment_by_name('rossdemo').artifact_location
    #     print("downloading artifacts from s3 bucket " + artifact_location)

    # Strip leading slashes for standard S3/MinIO key compatibility
    prefix_clean = f"{experiment_id}/{run_id}".lstrip("/")
    model_path = f"{prefix_clean}/artifacts/model/model.pkl"
    try:
        minioClient.fget_object("mlflow", model_path, "model.pkl")
    except Exception as e:
        print(f"Retrying with leading slash: {e}")
        minioClient.fget_object("mlflow", f"/{model_path}", "model.pkl")

    req_path = f"{prefix_clean}/artifacts/model/requirements.txt"
    try:
        minioClient.fget_object("mlflow", req_path, "requirements.txt")
    except Exception as e:
        print(f"Optional requirements.txt not found: {e}")
    
    # Download all pkl files from this location (including encoders)
    pkl_objects = list(minioClient.list_objects("mlflow", recursive=True, prefix=prefix_clean))
    if not pkl_objects:
        pkl_objects = list(minioClient.list_objects("mlflow", recursive=True, prefix=f"/{prefix_clean}"))

    for pkl_object in pkl_objects:
        pkl_object_name = pkl_object.object_name
        if pkl_object_name.endswith('.pkl'):
            print(f"Downloading artifact: {pkl_object_name}")
            minioClient.fget_object('mlflow', pkl_object_name, pkl_object_name.split("/")[-1])
      
    print("Download successful")
    return run_id

def build_push_image():
    container_location = os.environ["CONTAINER_DETAILS"]
    if os.environ.get("CONTAINER_REGISTRY") and os.environ["CONTAINER_REGISTRY"].find("docker.io") == -1:
        container_location = f"{os.environ['CONTAINER_REGISTRY']}/{container_location}"

    dockerfile_path = os.path.join(os.getcwd(), "Dockerfile")
    full_command = f"/kaniko/executor --context={os.getcwd()} --dockerfile={dockerfile_path} --verbosity=info --single-snapshot=true --destination={container_location}"
    print(f"Running Kaniko: {full_command}")
    subprocess.run(full_command, shell=True, check=True)

    # print(subprocess.check_output(['/kaniko/executor', '--context', '/workspace',  '--dockerfile', 'Dockerfile', '--destination', container_location]))


init()
download_artifacts()
build_push_image()

