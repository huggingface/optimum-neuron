import pytest

from optimum.neuron.utils import ecr


def test_ecr_image_uri():
    region = "eu-west-1"
    account_id = ecr.ACCOUNT_IDS[region]
    image_uri = ecr.image_uri(region=region, service_name="tgi", version="3.3.4")
    assert image_uri.startswith(f"{account_id}.dkr.ecr.{region}.amazonaws.com/huggingface-pytorch-tgi-inference:")
    # without params now
    image_uri = ecr.image_uri()
    assert image_uri is not None
    # only service name, sagemaker style
    image_uri = ecr.image_uri(service_name="huggingface-neuronx")
    assert image_uri is not None
    # only version
    image_uri = ecr.image_uri(version="3.3.4")
    assert image_uri is not None
    # invalid version
    image_uri = ecr.image_uri(version="ABCD")
    assert image_uri is None
    # not valid service name
    with pytest.raises(ValueError, match="Invalid service name"):
        ecr.image_uri(service_name="ABCD")
    # not valid region
    with pytest.raises(KeyError, match="ABCD"):
        ecr.image_uri(region="ABCD")
