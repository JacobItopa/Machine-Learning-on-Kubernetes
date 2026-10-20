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

auth_encoded = string.Template("$CONTAINER_REGISTRY_USER:$CONTAINER_REGISTRY_PASSWORD").substitute(os.environ)
os.environ["CONTAINER_REGISTRY_CREDS"] = base64.b64encode(auth_encoded.encode("ascii")).decode("ascii")

docker_auth = string.Template('{"auths":{"$CONTAINER_REGISTRY":{"auth":"$CONTAINER_REGISTRY_CREDS"}}}').substitute(os.environ)
print(docker_auth)
f = open("/kaniko/.docker/config.json", "w")
f.write(docker_auth)
f.close()

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

    try:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        model = client.get_registered_model(model_name)
        for lv in model.latest_versions:
            if str(lv.version) == str(model_version):
                exp_id = lv.source.split("/")[3] if len(lv.source.split("/")) > 3 else "1"
                return lv.run_id, exp_id, lv.source
    except Exception as e:
        print(f"MlflowClient failed ({e}), falling back to direct MLflow 2.0 REST API...")

    endpoints = [
        f"{host}/api/2.0/mlflow/model-versions/get?name={model_name}&version={model_version}",
        f"{host}/api/2.0/mlflow/registered-models/get?name={model_name}",
    ]

    for ep in endpoints:
        try:
            req = urllib.request.Request(ep)
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                if "model_version" in data:
                    mv = data["model_version"]
                    run_id = mv["run_id"]
                    source = mv.get("source", "")
                    exp_id = source.split("/")[3] if len(source.split("/")) > 3 else "1"
                    print(f"Resolved from REST model_version: run_id={run_id}, experiment_id={exp_id}")
                    return run_id, exp_id, source
                elif "registered_model" in data:
                    rm = data["registered_model"]
                    for lv in rm.get("latest_versions", []):
                        if str(lv.get("version")) == str(model_version):
                            run_id = lv["run_id"]
                            source = lv.get("source", "")
                            exp_id = source.split("/")[3] if len(source.split("/")) > 3 else "1"
                            print(f"Resolved from REST registered_model: run_id={run_id}, experiment_id={exp_id}")
                            return run_id, exp_id, source
        except Exception as err:
            print(f"REST endpoint {ep} returned: {err}")

    raise RuntimeError(f"Could not retrieve metadata for model '{model_name}' version '{model_version}' from {host}")


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
    container_location = string.Template("$CONTAINER_REGISTRY/$CONTAINER_DETAILS").substitute(os.environ)
    
    #For docker repo, do not include the registry domain name in container location
    if os.environ["CONTAINER_REGISTRY"].find("docker.io") != -1:
        container_location= os.environ["CONTAINER_DETAILS"]
        
    full_command = "/kaniko/executor --context=" + os.getcwd() + " --dockerfile=Dockerfile --verbosity=debug --cache=true --single-snapshot=true --destination=" + container_location
    print(full_command)
    process = subprocess.run(full_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    print(process.stdout)
    print(process.stderr)

    # print(subprocess.check_output(['/kaniko/executor', '--context', '/workspace',  '--dockerfile', 'Dockerfile', '--destination', container_location]))


init()
download_artifacts()
build_push_image()

