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

version:
	@echo ${VERSION}

PACKAGE_DIST = dist/optimum_neuron-$(VERSION).tar.gz
PACKAGE_WHEEL = dist/optimum_neuron-$(VERSION)-py3-none-any.whl
PACKAGE_PYTHON_FILES = $(call rwildcard, optimum/*.py)
PACKAGE_FILES = $(PACKAGE_PYTHON_FILES)  \
				pyproject.toml \
				README.md \
				MANIFEST.in

# Package build recipe
$(PACKAGE_DIST) $(PACKAGE_WHEEL): $(PACKAGE_FILES)
	python -m build

# Creates example scripts from Transformers
transformers_examples:
	rm -f examples/**/*.py
	python tools/create_examples_from_transformers.py --version $(VERSION) examples

# Run code quality checks
style_check:
	ruff check .
	ruff format . --diff

style:
	ruff check . --fix
	ruff format .

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

# Docker images

optimum-neuron-vllm: docker/vllm/Dockerfile ${PACKAGE_FILES}
	docker build --rm \
		-f docker/vllm/Dockerfile \
		-t optimum-neuron-vllm:${VERSION} .
	docker tag optimum-neuron-vllm:${VERSION} optimum-neuron-vllm:latest
