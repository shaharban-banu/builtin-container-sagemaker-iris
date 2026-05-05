import sagemaker
from sagemaker.sklearn.model import SKLearnModel


role="arn:aws:iam::285407030131:role/sagemaker_execution_role"

model=SKLearnModel(
    model_data="s3://sagemaker-bucket-11/model.tar.gz",
    role=role,
    entry_point="inference.py",
    framework_version="1.2-1"
)

endpoint_name="iris-endpoint"

predictor=model.deploy(
    instance_type="ml.t2.medium",
    initial_instance_count=1,
    endpoint_name=endpoint_name
)