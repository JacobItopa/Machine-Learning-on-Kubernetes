import os
import socket
import sys
import time
import subprocess


# Ensure JAVA_HOME is set in modern JupyterLab
if "JAVA_HOME" not in os.environ and os.path.exists("/opt/conda"):
    os.environ["JAVA_HOME"] = "/opt/conda"

# Get all installed packages as installed_packages
reqs = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'])
installed_packages = [r.decode().split('==')[0] for r in reqs.split()]


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])


def get_cluster_name(index=0, from_env=False):
    # Get the cluster name from env var, spark-info.txt, or default
    if os.path.exists('spark-info.txt') and not from_env:
        with open('spark-info.txt', 'r') as sparkinfo:
            lines = sparkinfo.readlines()
            if lines:
                spark_cluster_name = lines[0 + (index * 2)].replace("\n", "").strip()
                os.environ['SPARK_CLUSTER'] = spark_cluster_name
                return spark_cluster_name

    cluster = os.environ.get('SPARK_CLUSTER', 'spark-cluster-sample')
    if cluster.startswith('spark-cluster-') or cluster.startswith('local'):
        return cluster
    return f"spark-cluster-{cluster}"


def get_driver_host_ip():
    # Returns the IP address where this process is running on
    try:
        ip_address = socket.gethostbyname(socket.gethostname())
    except Exception:
        ip_address = "127.0.0.1"
    return ip_address


def init_environment(submit_args):
    # Prepares Spark related configuration through PYSPARK_SUBMIT_ARGS env variable
    cluster_name = get_cluster_name()
    # Default to local[2] instead of local[*] to limit concurrent S3A writers inside the container
    master_target = os.environ.get('SPARK_MASTER', 'local[2]')
    
    # If the user explicitly sets SPARK_MASTER=cluster, connect to standalone master
    if master_target == "cluster":
        master_url = f"spark://{cluster_name}:7077"
    else:
        master_url = master_target

    # Ensure sufficient driver memory for 5.8M row dataset (default is 1g)
    if "--driver-memory" not in submit_args and "spark.driver.memory" not in submit_args:
        submit_args = f"--driver-memory 4g {submit_args}"

    # PySpark 3.4/3.5 bundles Hadoop 3.3.x runtime libraries.
    # Legacy hadoop-aws:3.2.0 causes NoSuchMethodError on SemaphoredDelegatingExecutor during S3A write operations.
    # Automatically upgrade hadoop-aws:3.2.0 to 3.3.4 + matching aws-java-sdk-bundle
    if "hadoop-aws:3.2.0" in submit_args:
        submit_args = submit_args.replace(
            "org.apache.hadoop:hadoop-aws:3.2.0",
            "org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262"
        )

    os.environ['PYSPARK_SUBMIT_ARGS'] = f"{submit_args} --master {master_url} pyspark-shell "


def configure_minio_s3a(spark):
    """
    Configures S3A settings for self-hosted MinIO to prevent:
    - ClassNotFoundException (missing AWS SDK v2 classes)
    - NumberFormatException (30s, 60s, 24h string values in Hadoop 3.2)
    - OutOfMemoryError: unable to create native thread (capping S3A thread pool)
    - Heap exhaustion during multipart upload (buffering to disk instead of RAM)
    """
    try:
        hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()
        hadoop_conf.set("fs.s3a.endpoint", "http://minio-ml-workshop:9000")
        hadoop_conf.set("fs.s3a.access.key", "minio")
        hadoop_conf.set("fs.s3a.secret.key", "minio123")
        hadoop_conf.set("fs.s3a.path.style.access", "true")
        hadoop_conf.set("fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        hadoop_conf.set("fs.s3a.connection.ssl.enabled", "false")
        hadoop_conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
        
        # Prevent thread explosion (pthread_create EAGAIN) in Docker/K8s containers
        hadoop_conf.set("fs.s3a.threads.max", "10")
        hadoop_conf.set("fs.s3a.connection.maximum", "20")
        
        # Buffer uploads to local container disk instead of consuming JVM heap memory
        hadoop_conf.set("fs.s3a.fast.upload.buffer", "disk")
        hadoop_conf.set("fs.s3a.multipart.size", "67108864")
        
        # Raw numeric overrides for Hadoop compatibility
        hadoop_conf.set("fs.s3a.connection.timeout", "30000")
        hadoop_conf.set("fs.s3a.connection.establish.timeout", "30000")
        hadoop_conf.set("fs.s3a.connection.request.timeout", "30000")
        hadoop_conf.set("fs.s3a.threads.keepalivetime", "60")
        hadoop_conf.set("fs.s3a.multipart.purge", "false")
        hadoop_conf.set("fs.s3a.multipart.purge.age", "86400")
    except Exception as e:
        print(f"Warning: Could not configure Hadoop S3A settings: {e}")


def getOrCreateSparkSession(application_name, submit_args="", log_level="INFO"):
    # Creates a Spark session from a given Spark application name and optional submit_args

    if "pyspark" not in installed_packages:
        install("pyspark")

    from pyspark.sql import SparkSession

    print("Initializing environment variables for Spark")
    init_environment(submit_args)
    
    # Use 127.0.0.1 for local mode, or driver IP for cluster mode
    master_target = os.environ.get('SPARK_MASTER', 'local[*]')
    bind_ip = "127.0.0.1" if master_target.startswith("local") else get_driver_host_ip()

    sparkSessionBuilder = SparkSession.builder \
        .config("spark.driver.host", bind_ip) \
        .config("spark.driver.bindAddress", bind_ip) \
        .appName(application_name)

    print("Creating a spark session...")
    spark = sparkSessionBuilder.getOrCreate()
    spark.sparkContext.setLogLevel(log_level)
    
    # Apply MinIO S3A configuration fixes automatically
    configure_minio_s3a(spark)
    
    print("Spark session created")
    
    cluster_name, app_url = get_app_url(spark.sparkContext.applicationId)
    try:
        from IPython.core.display import display, HTML
        display(HTML(f"""
            <div>
                <br/>
                <p><b>Spark Context</b></p>
                <dl>
                  <dt>Cluster Name</dt>
                    <dd><code>{cluster_name}</code></dd>
                  <dt>Version</dt>
                    <dd><code>v{spark.sparkContext.version}</code></dd>
                  <dt>Master</dt>
                    <dd><code>{spark.sparkContext.master}</code></dd>
                  <dt>App Id</dt>
                    <dd><code>{spark.sparkContext.applicationId}</code></dd>
                  <dt>App Name</dt>
                    <dd><code>{spark.sparkContext.appName}</code></dd>
                  <dt>Driver IP</dt>
                    <dd><code>{bind_ip}</code></dd>
                </dl>
            </div>
        """))
    except Exception:
        pass
    
    return spark


def get_app_url(app_id):
    cluster_name = get_cluster_name()
    app_url = f"http://{cluster_name}/app/?appId={app_id}"
    return cluster_name, app_url


def wait(predicate, timeout):
    mustend = time.time() + timeout
    time.sleep(5)
    while time.time() < mustend:
        try:
            if predicate(1):
                return True
        except Exception as ex:
            print(ex)
        time.sleep(5)
    return False


def get_k8s_info():
    server = "https://" + os.environ.get("KUBERNETES_SERVICE_HOST", "kubernetes.default.svc") + ":" + os.environ.get("KUBERNETES_SERVICE_PORT", "443")
    token = ""
    token_path = '/var/run/secrets/kubernetes.io/serviceaccount/token'
    if os.path.exists(token_path):
        with open(token_path, 'r') as f:
            token = f.read()

    namespace = "ml-workshop"
    ns_path = '/var/run/secrets/kubernetes.io/serviceaccount/namespace'
    if os.path.exists(ns_path):
        with open(ns_path, 'r') as f:
            namespace = f.read()

    return server, token, namespace