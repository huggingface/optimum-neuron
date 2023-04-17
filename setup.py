import re

from setuptools import find_namespace_packages, setup


# Ensure we match the version set in optimum/neuron/version.py
try:
    filepath = "optimum/neuron/version.py"
    with open(filepath) as version_file:
        (__version__,) = re.findall('__version__ = "(.*)"', version_file.read())
except Exception as error:
    assert False, "Error: Could not open '%s' due %s\n" % (filepath, error)


INSTALL_REQUIRES = [
    "transformers >= 4.28.0",
    "optimum",
]

TESTS_REQUIRE = [
    "pytest",
    "psutil",
    "parameterized",
    "GitPython",
    "sentencepiece",
    "datasets",
]

QUALITY_REQUIRES = [
    "black",
    "ruff",
    "isort",
    "hf_doc_builder @ git+https://github.com/huggingface/doc-builder.git",
]

EXTRAS_REQUIRE = {
    "tests": TESTS_REQUIRE,
    "quality": QUALITY_REQUIRES,
    "neuron": ["neuron-cc[tensorflow]", "torch-neuron", "protobuf==3.20.2", "torchvision"],
    "neuronx": ["neuronx-cc==2.*", "torch-neuronx", "torchvision"],
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
    include_package_data=True,
    zip_safe=False,
    entry_points={"console_scripts": ["optimum-cli=optimum.commands.optimum_cli:main"]},
)
