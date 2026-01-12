#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
This script lists images in HuggingFace neuron ECR repositories.
"""

import argparse
import base64
import os
import subprocess
import sys
from typing import Optional

import boto3
from botocore.exceptions import ClientError


HUGGINGFACE_PUBLISHING_AWS_ACCOUNT_ID = "763104351884"

ECR_REPOSITORIES = {
    "inference": "huggingface-pytorch-inference-neuronx",
    "training": "huggingface-pytorch-training-neuronx",
    "vllm": "huggingface-vllm-inference-neuronx",
}


def login_to_ecr(ecr_client, region: str) -> None:
    """Log in to HuggingFace public ECR."""
    ecr_url = f"{HUGGINGFACE_PUBLISHING_AWS_ACCOUNT_ID}.dkr.ecr.{region}.amazonaws.com"
    print(f"Logging in to Amazon ECR at {ecr_url}...")

    try:
        response = ecr_client.get_authorization_token(registryIds=[HUGGINGFACE_PUBLISHING_AWS_ACCOUNT_ID])

        auth_data = response["authorizationData"][0]
        auth_token = base64.b64decode(auth_data["authorizationToken"]).decode()
        username, password = auth_token.split(":")

        # Use docker login
        process = subprocess.Popen(
            ["docker", "login", "--username", username, "--password-stdin", ecr_url],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stderr = process.communicate(input=password.encode())[1]

        if process.returncode != 0:
            print(f"Warning: Docker login failed: {stderr.decode()}")
        else:
            print("Successfully logged in to ECR")

    except ClientError as e:
        print(f"Error: Unable to get ECR authorization token. {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Warning: Docker login encountered an issue: {e}")


def get_ecr_images(ecr_client, repository_name: str, region: str, limit: Optional[int] = None) -> list:
    """List all images in the ECR repository, filtered by images with tags and sorted by tag."""

    try:
        paginator = ecr_client.get_paginator("list_images")
        page_iterator = paginator.paginate(
            repositoryName=repository_name, registryId=HUGGINGFACE_PUBLISHING_AWS_ACCOUNT_ID
        )

        # Collect all images with tags
        tagged_images = []
        for page in page_iterator:
            for image_id in page.get("imageIds", []):
                # Only include images that have an imageTag
                if "imageTag" in image_id:
                    image_digest = image_id.get("imageDigest", "")
                    image_tag = image_id.get("imageTag")
                    tagged_images.append((image_digest, image_tag))

        # Sort by imageTag
        tagged_images.sort(key=lambda x: x[1])

        # Apply limit if specified (show last n tags)
        total_count = len(tagged_images)
        if limit is not None and limit > 0:
            tagged_images = tagged_images[-limit:]

        return tagged_images, total_count

    except ClientError as e:
        print(f"Error: Unable to list images. {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="List all images in HuggingFace public ECR repositories.")
    parser.add_argument(
        "repository_type",
        nargs="?",
        choices=ECR_REPOSITORIES.keys(),
        default=None,
    )
    parser.add_argument(
        "--region",
        nargs="?",
        default=os.environ.get("AWS_REGION", os.environ.get("AWS_DEFAULT_REGION", "us-west-2")),
        help="AWS region (default: AWS_REGION or AWS_DEFAULT_REGION or us-west-2)",
    )
    parser.add_argument(
        "-n", "--limit", type=int, default=1, help="Limit output to the last N tags (sorted alphabetically)"
    )

    args = parser.parse_args()

    # Create AWS client
    ecr_client = boto3.client("ecr", region_name=args.region)

    # Log in to ECR
    login_to_ecr(ecr_client, args.region)

    # List images
    repository_types = ECR_REPOSITORIES.keys() if args.repository_type is None else [args.repository_type]
    for repository_type in repository_types:
        images = get_ecr_images(ecr_client, ECR_REPOSITORIES[repository_type], args.region, args.limit)
        print(f"\nRepository: {repository_type} ({ECR_REPOSITORIES[repository_type]})")
        print(f"Showing {len(images[0])} of {images[1]} images:")
        for image in images[0]:
            print(f"  Tag: {image[1]}")


if __name__ == "__main__":
    main()
