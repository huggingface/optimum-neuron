#  Copyright 2022 The HuggingFace Team. All rights reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
SHELL := /bin/bash
CURRENT_DIR = $(shell pwd)
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-neuron.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))

.PHONY:	build_dist style style_check clean

clean:
	rm -rf dist

rwildcard=$(wildcard $1) $(foreach d,$1,$(call rwildcard,$(addsuffix /$(notdir $d),$(wildcard $(dir $d)*))))

VERSION := $(shell gawk 'match($$0, /__version__ = "(.*)"/, a) {print a[1]}' optimum/neuron/version.py)

PACKAGE_DIST = dist/optimum_neuron-$(VERSION).tar.gz
PACKAGE_WHEEL = dist/optimum_neuron-$(VERSION)-py3-none-any.whl
PACKAGE_PYTHON_FILES = $(call rwildcard, optimum/*.py)
PACKAGE_FILES = $(PACKAGE_PYTHON_FILES)  \
				setup.py \
				setup.cfg \
				pyproject.toml \
				README.md \
				MANIFEST.in

# Package build recipe
$(PACKAGE_DIST) $(PACKAGE_WHEEL): $(PACKAGE_FILES)
	python -m build

neuronx-tgi: $(PACKAGE_DIST)
	docker build --rm -f text-generation-inference/Dockerfile \
	             --build-arg VERSION=$(VERSION) \
				 -t neuronx-tgi:$(VERSION) .
	docker tag neuronx-tgi:$(VERSION) neuronx-tgi:latest

neuronx-tgi-sagemaker: $(PACKAGE_DIST)
	docker build --rm -f text-generation-inference/Dockerfile \
	             --build-arg VERSION=$(VERSION) \
				 --target sagemaker \
				 -t neuronx-tgi:$(VERSION) .

# Creates example scripts from Transformers
transformers_examples:
	rm -f examples/**/*.py
	python tools/create_examples_from_transformers.py --version $(VERSION) examples

# Run code quality checks
style_check:
	black --check .
	ruff check .

style:
	black .
	ruff check . --fix

# Utilities to release to PyPi
build_dist_install_tools:
	python -m pip install build
	python -m pip install twine

build_dist: ${PACKAGE_DIST} ${PACKAGE_WHEEL}

pypi_upload: ${PACKAGE_DIST} ${PACKAGE_WHEEL}
	python -m twine upload ${PACKAGE_DIST} ${PACKAGE_WHEEL}

# Tests

test_installs:
	python -m pip install .[tests]
	python -m pip install git+https://github.com/huggingface/transformers.git

# Stand-alone TGI server for unit tests outside of TGI container
tgi_server:
	python -m pip install -r text-generation-inference/server/build-requirements.txt
	make -C text-generation-inference/server clean
	VERSION=${VERSION} make -C text-generation-inference/server gen-server

tgi_test: tgi_server
	python -m pip install .[neuronx]
	python -m pip install -r text-generation-inference/tests/requirements.txt
	find text-generation-inference -name "text_generation_server-$(VERSION)-py3-none-any.whl" \
	                               -exec python -m pip install --force-reinstall {} \;
	python -m pytest -sv text-generation-inference/tests -k server

tgi_docker_test: neuronx-tgi
	python -m pip install -r text-generation-inference/tests/requirements.txt
	python -m pytest -sv text-generation-inference/tests -k integration
