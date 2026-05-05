import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import time
import boto3

role="arn:aws:iam::285407030131:role/sagemaker_execution_role"

endpoint_name="iris-endpoint"
config_name = f"{endpoint_name}-config-{int(time.time())}"

boto_session = boto3.Session(region_name="ap-south-1")
session = sagemaker.Session(boto_session=boto_session)


#endpoint_name = f"iris-endpoint-{int(time.time())}"

model = SKLearnModel(
    model_data="s3://sagemaker-bucket-11/model.tar.gz",
    role=role,
    entry_point="inference.py",  
    framework_version="1.2-1",
    sagemaker_session=session
)

# 🔍 Check if endpoint exists
sm_client = boto3.client("sagemaker", region_name="ap-south-1")

try:
    sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_exists = True
except:
    endpoint_exists = False

# 🚀 Deploy logic
if endpoint_exists:
    print("🔄 Updating existing endpoint...")
    predictor = model.deploy(
        instance_type="ml.t2.medium",
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        endpoint_config_name=config_name,
        update_endpoint=True
    )
else:
    print("🚀 Creating new endpoint...")
    predictor = model.deploy(
        instance_type="ml.t2.medium",
        initial_instance_count=1,
        endpoint_name=endpoint_name,
        endpoint_config_name=config_name
    )

print("✅ Deployment complete")

# sm_client=boto3.client("sagemaker")
# sm_session=sagemaker.Session()

# #config_name=f"{endpoint_name}--{int(time.time())}"

# def endpoint_exist(name):
#     try:
#         sm_client.describe_endpoint(EndpointName=name)
#         return True
#     except:
#         return False

# model=SKLearnModel(
#     model_data="s3://sagemaker-bucket-11/model.tar.gz",
#     role=role,
#     entry_point="inference.py",
#     framework_version="1.2-1"
# )

# if endpoint_exist(endpoint_name):
#     print("updating existing endpoint...")

#     model_name=f"iris-model--{int(time.time())}"
#     container=model.prepare_container_def()

#     sm_client.creat
# else:
#     print("creating new endpoint.....")
#     predictor=model.deploy(
#         instance_type="ml.t2.medium",
#         initial_instance_count=1,
#         endpoint_name=endpoint_name
#     )