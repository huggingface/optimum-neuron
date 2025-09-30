import re

import boto3


# pulled from https://github.com/aws/sagemaker-python-sdk/blob/master/src/sagemaker/image_uri_config/huggingface-llm-neuronx.json
ACCOUNT_IDS = {
    "ap-northeast-1": "763104351884",
    "ap-south-1": "763104351884",
    "ap-south-2": "772153158452",
    "ap-southeast-1": "763104351884",
    "ap-southeast-2": "763104351884",
    "ap-southeast-4": "457447274322",
    "ap-southeast-5": "550225433462",
    "ap-southeast-7": "590183813437",
    "cn-north-1": "727897471807",
    "cn-northwest-1": "727897471807",
    "eu-central-1": "763104351884",
    "eu-central-2": "380420809688",
    "eu-south-2": "503227376785",
    "eu-west-1": "763104351884",
    "eu-west-3": "763104351884",
    "il-central-1": "780543022126",
    "mx-central-1": "637423239942",
    "sa-east-1": "763104351884",
    "us-east-1": "763104351884",
    "us-east-2": "763104351884",
    "us-gov-east-1": "446045086412",
    "us-gov-west-1": "442386744353",
    "us-west-2": "763104351884",
    "ca-west-1": "204538143572",
}

TGI_REPOSITORY_NAME = "huggingface-pytorch-tgi-inference"

# Later on, other services might be added. The format is {service_name: repository_name}
IMAGE_SERVICES = {
    "tgi": TGI_REPOSITORY_NAME,
    "huggingface-neuronx": TGI_REPOSITORY_NAME,
}


def check_tag(tag, expected_version):
    """
    Verify that the expected_version appears after 'optimum' in the tag string.
    """
    pattern = rf"optimum[.-]?{re.escape(expected_version)}"
    match = re.search(pattern, tag)
    return match is not None


def image_uri(
    service_name: str = "tgi",
    region: str = None,
    version: str = None,
):
    """Get the image URI for the given service name, region and version.
    This can be used to get the image URI for a service provided by one of the Optimum Neuron containers.
    The service name can be "tgi" or "huggingface-neuronx" (as in get_huggingface_llm_image_uri from the Sagemaker SDK).
    The image retrieved can be newer than the one reported by the Sagemaker SDK.

    Args:
        service_name: The name of the service to get the image URI for.
        region: The region to get the image URI for. If not provided, the region will be inferred from the boto3 session.
        version: The version to get the image URI for. If not provided, the latest tag will be used.

    Returns:
        The image URI for the given service name, region and version.
    """
    ecr_client = boto3.client("ecr")
    if region is None:
        # use default region from boto3
        region = ecr_client.meta.region_name
    repository_id = ACCOUNT_IDS[region]
    if service_name not in IMAGE_SERVICES:
        raise ValueError(f"Invalid service name: {service_name}")
    repository_name = IMAGE_SERVICES[service_name]
    try:
        images = ecr_client.list_images(repositoryName=repository_name, registryId=repository_id)["imageIds"]
    except boto3.exceptions.botocore.exceptions.NoCredentialsError as e:
        message = (
            "An error occurred while listing images from the ECR repository. Please check you have exported "
            + " the AWS credentials. Original error: "
            + str(e)
        )
        raise ValueError(message)

    neuronx_images = [image for image in images if "neuronx" in image["imageTag"]]
    if version is not None:
        neuronx_images = [image for image in neuronx_images if check_tag(image["imageTag"], version)]
        if len(neuronx_images) == 0:
            return None
        tag = neuronx_images[-1]["imageTag"]
    else:
        # Get latest tag
        neuronx_images.sort(key=lambda x: x["imageTag"])
        tag = neuronx_images[-1]["imageTag"]

    return f"{repository_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{tag}"
