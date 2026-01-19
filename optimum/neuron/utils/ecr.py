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

VLLM_REPOSITORY_NAME = "huggingface-vllm-inference-neuronx"
INFERENCE_REPOSITORY_NAME = "huggingface-pytorch-inference-neuronx"
TRAINING_REPOSITORY_NAME = "huggingface-pytorch-training-neuronx"

# Later on, other services might be added. The format is {service_name: repository_name}
IMAGE_SERVICES = {
    "inference": INFERENCE_REPOSITORY_NAME,
    "training": TRAINING_REPOSITORY_NAME,
    "vllm": VLLM_REPOSITORY_NAME,
}

# This dictionary contains the pattern templates for the version pattern for each repository. The
# "platform_version" will be replaced with the platform version when provided.
TAG_PATTERNS = {
    INFERENCE_REPOSITORY_NAME: r"transformers{platform_version}",
    TRAINING_REPOSITORY_NAME: r"transformers{platform_version}",
    VLLM_REPOSITORY_NAME: r"optimum{platform_version}",
}


def check_tag(pattern, tag: list[dict], platform_version: str = None) -> list[str]:
    """Return all the tags that match the version pattern

    Args:
        pattern_template: the pattern template to match
        tag: list of dicts with the image tag
        platform_version: the platform version to match. If not provided, all versions will be matched.

    Returns:
        list of tags that match the version pattern
    """
    if platform_version is None:
        platform_version = ".*"  # all versions
    else:
        platform_version = re.escape(platform_version)
    version_pattern = re.compile(pattern.format(platform_version=platform_version))
    return [t["imageTag"] for t in tag if version_pattern.search(t["imageTag"]) is not None]


def image_uri(
    service_name: str = "vllm",
    region: str = None,
    version: str = None,
):
    """Get the image URI for the given service name, region and version.
    This can be used to get the image URI for a service provided by one of the Optimum Neuron containers.
    The image retrieved can be newer than the one reported by the Sagemaker SDK.

    Note: when looking for a specific version, use the optimum-neuron version for the vLLM image and the transformers version for
    inference and training images.

    Args:
        service_name: The name of the service to get the image URI for.
        region: The region to get the image URI for. If not provided, the region will be inferred from the boto3 session.
        version: The version to get the image URI for. If not provided, the latest tag will be used.

    Returns:
        The image URI for the given service name, region and version.
    """
    if region is None:
        # Try first to use default region from boto3
        session = boto3.session.Session()
        region = session.region_name
    if region is None:
        raise ValueError("Region must be specified either explicitly or in the environment using AWS_DEFAULT_REGION.")
    if region not in ACCOUNT_IDS:
        raise KeyError(f"Invalid region: {region}")
    repository_id = ACCOUNT_IDS[region]
    if service_name not in IMAGE_SERVICES:
        raise ValueError(f"Invalid service name: {service_name}")
    repository_name = IMAGE_SERVICES[service_name]
    ecr_client = boto3.client("ecr", region_name=region)
    try:
        images = ecr_client.list_images(repositoryName=repository_name, registryId=repository_id)["imageIds"]
    except boto3.exceptions.botocore.exceptions.NoCredentialsError as e:
        message = (
            "An error occurred while listing images from the ECR repository. Please check you have set "
            + " the required AWS credentials. Original error: "
            + str(e)
        )
        raise ValueError(message)
    # Remove images without an image Tag (these are untagged images)
    images = [img for img in images if "imageTag" in img]
    repository_pattern = TAG_PATTERNS[repository_name]
    tags = check_tag(repository_pattern, images, version)
    if len(tags) == 0:
        return None
    # sorting will put the latest version at the end
    tags.sort()
    tag = tags[-1]
    return f"{repository_id}.dkr.ecr.{region}.amazonaws.com/{repository_name}:{tag}"
