import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/neuron/version.py
filepath = "optimum/neuron/version.py"
try:
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


INSTALL_REQUIRES = [
    "transformers == 4.41.1",
    "accelerate == 0.29.2",
    "optimum ~= 1.20.0",
    "huggingface_hub >= 0.20.1",
    "numpy>=1.22.2, <=1.25.2",
    "protobuf<4",
]

TESTS_REQUIRE = [
    "pytest <= 8.0.0",
    "psutil",
    "parameterized",
    "GitPython",
    "sentencepiece",
    "datasets",
    "sacremoses",
    "diffusers >= 0.26.1",
    "safetensors",
    "sentence-transformers >= 2.2.0",
    "peft",
    "compel",
    "rjieba",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "isort",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRES,
    "neuron": [
        "wheel",
        "torch-neuron==1.13.1.2.9.74.0",
        "torch==1.13.1.*",
        "neuron-cc[tensorflow]==1.22.0.0",
        "protobuf",
        "torchvision",
    ],
    "neuronx": [
        "wheel",
        "neuronx-cc==2.13.66.0",
        "torch-neuronx==2.1.2.2.1.0",
        "transformers-neuronx==0.10.0.21",
        "torch==2.1.2.*",
        "torchvision==0.16.*",
        "neuronx_distributed==0.7.0",
    ],
    "diffusers": ["diffusers ~= 0.26.1", "peft"],
    "sentence-transformers": ["sentence-transformers >= 2.2.0"],
}

setup(
    name="optimum-neuron",
    version=__version__,
    description=(
        "Optimum Neuron is the interface between the Hugging Face Transformers and Diffusers libraries and AWS "
        "Tranium and Inferentia accelerators. It provides a set of tools enabling easy model loading, training and "
        "inference on single and multiple neuron core settings for different downstream tasks."
    ),
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="transformers, diffusers, mixed-precision training, fine-tuning, inference, tranium, inferentia, aws",
    url="https://huggingface.co/hardware/aws",
    author="HuggingFace Inc. Special Ops Team",
    author_email="hardware@huggingface.co",
    license="Apache",
    packages=find_namespace_packages(include=["optimum*"]),
    install_requires=INSTALL_REQUIRES,
    extras_require=EXTRAS_REQUIRE,
    dependency_links=["https://pip.repos.neuron.amazonaws.com"],
    include_package_data=True,
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "optimum-cli=optimum.commands.optimum_cli:main",
            "neuron_parallel_compile=optimum.neuron.utils.neuron_parallel_compile:main",
        ]
    },
)
