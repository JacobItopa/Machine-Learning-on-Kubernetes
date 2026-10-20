import os
import sys

def init_mlflow(
    experiment_name="Default_Experiment",
    tracking_uri=None,
    minio_endpoint=None,
    access_key=None,
    secret_key=None,
    autolog=True
):
    """
    Initializes MLflow tracking client and configures S3/MinIO environment variables
    for artifact (model) storage.
    
    Usage:
        import mlflow_util
        mlflow_util.init_mlflow("FlightsDelay-mluser")
    """
    # Ensure required packages for MLflow S3 artifact tracking are installed
    # Pin mlflow==2.11.3 to match the cluster's MLflow server version and prevent 404 on newer endpoints
    for pkg, pkg_spec in [("mlflow", "mlflow==2.11.3"), ("boto3", "boto3")]:
        try:
            __import__(pkg)
        except ImportError:
            import subprocess
            print(f"Installing {pkg_spec} for S3/MinIO artifact logging...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg_spec])
    import mlflow

    # 1. Resolve host and tracking URI
    host = tracking_uri or os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5500")
    mlflow.set_tracking_uri(host)

    # 2. Resolve MinIO / S3 endpoint and credentials
    s3_endpoint = (
        minio_endpoint
        or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")
        or "http://minio-ml-workshop:9000"
    )
    s3_key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    s3_secret = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")
    s3_region = os.environ.get("AWS_REGION", "us-east-1")
    s3_bucket = os.environ.get("AWS_BUCKET_NAME", "mlflow")

    os.environ["MLFLOW_S3_ENDPOINT_URL"] = s3_endpoint
    os.environ["AWS_ACCESS_KEY_ID"] = s3_key
    os.environ["AWS_SECRET_ACCESS_KEY"] = s3_secret
    os.environ["AWS_REGION"] = s3_region
    os.environ["AWS_BUCKET_NAME"] = s3_bucket

    # Safely close any previous active run that was left dangling
    try:
        if mlflow.active_run():
            mlflow.end_run()
    except Exception:
        pass

    # 3. Set the experiment
    if experiment_name:
        mlflow.set_experiment(experiment_name)

    # 4. Enable scikit-learn autologging
    if autolog:
        mlflow.sklearn.autolog(log_input_examples=True)

    print(f"MLflow initialized successfully:")
    print(f"  • Tracking Server: {host}")
    print(f"  • Artifact Store:  {s3_endpoint} (Bucket: {s3_bucket})")
    print(f"  • Experiment:      {experiment_name}")
    return mlflow


def get_s3_fs(minio_endpoint=None, access_key=None, secret_key=None):
    """
    Returns a configured s3fs.S3FileSystem instance authenticated for MinIO.
    Avoids the common 'anon=True' read permission pitfall.
    """
    try:
        import s3fs
    except ImportError:
        import subprocess
        print("Installing s3fs...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "s3fs"])
        import s3fs

    endpoint = (
        minio_endpoint
        or os.environ.get("MLFLOW_S3_ENDPOINT_URL")
        or os.environ.get("MINIO_ENDPOINT")
        or "http://minio-ml-workshop:9000"
    )
    key = access_key or os.environ.get("AWS_ACCESS_KEY_ID", "minio")
    secret = secret_key or os.environ.get("AWS_SECRET_ACCESS_KEY", "minio123")

    return s3fs.S3FileSystem(
        anon=False,
        key=key,
        secret=secret,
        client_kwargs=dict(endpoint_url=endpoint),
    )


def read_parquet_from_s3(s3_path, minio_endpoint=None):
    """
    Helper to read Parquet data directly from MinIO into a Pandas DataFrame.
    
    Usage:
        df = mlflow_util.read_parquet_from_s3("s3://flights-data/flights-clean")
    """
    fs = get_s3_fs(minio_endpoint=minio_endpoint)
    # Strip protocol prefix (s3:// or s3a://) to match s3fs and avoid PyArrow base dir mismatch
    clean_path = s3_path.replace("s3://", "").replace("s3a://", "").lstrip("/")
    print(f"Reading dataset from {clean_path} via MinIO...")

    try:
        import pandas as pd
        df = pd.read_parquet(clean_path, filesystem=fs)
    except Exception:
        try:
            import pyarrow.parquet as pq
        except ImportError:
            import subprocess
            print("Installing pyarrow...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
            import pyarrow.parquet as pq
        df = pq.read_table(clean_path, filesystem=fs).to_pandas()

    print(f"Loaded {len(df):,} records into Pandas DataFrame.")
    return df
