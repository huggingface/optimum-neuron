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
DEFAULT_CLONE_URL := https://github.com/huggingface/optimum-habana.git
# If CLONE_URL is empty, revert to DEFAULT_CLONE_URL
REAL_CLONE_URL = $(if $(CLONE_URL),$(CLONE_URL),$(DEFAULT_CLONE_URL))


.PHONY:	style test

# Run code quality checks
style_check:
	black --check .
	ruff .

style:
	black .
	ruff . --fix

# Run unit and integration tests
fast_tests:
	python -m pip install .[tests]
	python -m pytest tests/test_gaudi_configuration.py tests/test_trainer_distributed.py tests/test_trainer.py tests/test_trainer_seq2seq.py

# Run unit and integration tests related to Diffusers
fast_tests_diffusers:
	python -m pip install .[tests]
	python -m pytest tests/test_diffusers.py

# Run single-card non-regression tests
slow_tests_1x: test_installs
	python -m pytest tests/test_examples.py -v -s -k "single_card"

# Run multi-card non-regression tests
slow_tests_8x: test_installs
	python -m pytest tests/test_examples.py -v -s -k "multi_card"

# Run DeepSpeed non-regression tests
slow_tests_deepspeed: test_installs
	python -m pip install git+https://github.com/HabanaAI/DeepSpeed.git@1.8.0
	python -m pytest tests/test_examples.py -v -s -k "deepspeed"

slow_tests_diffusers: test_installs
	python -m pip install git+https://github.com/huggingface/diffusers.git
	python -m pip install ftfy
	python -m pytest tests/test_diffusers.py -v -s -k "test_no_"

# Check if examples are up to date with the Transformers library
example_diff_tests: test_installs
	python -m pytest tests/test_examples_match_transformers.py

# Utilities to release to PyPi
build_dist_install_tools:
	python -m pip install build
	python -m pip install twine

build_dist:
	rm -fr build
	rm -fr dist
	python -m build

pypi_upload: build_dist
	python -m twine upload dist/*

build_doc_docker_image:
	docker build -t doc_maker --build-arg commit_sha=$(COMMIT_SHA_SUBPACKAGE) --build-arg clone_url=$(REAL_CLONE_URL) ./docs

doc: build_doc_docker_image
	@test -n "$(BUILD_DIR)" || (echo "BUILD_DIR is empty." ; exit 1)
	@test -n "$(VERSION)" || (echo "VERSION is empty." ; exit 1)
	docker run -v $(CURRENT_DIR):/doc_folder --workdir=/doc_folder doc_maker \
	doc-builder build optimum.habana /optimum-habana/docs/source/ \
		--build_dir $(BUILD_DIR) \
		--version $(VERSION) \
		--version_tag_suffix "" \
		--html \
		--clean

clean:
	find . -name "habana_log.livealloc.log_*" -type f -delete
	find . -name .lock -type f -delete
	find . -name .graph_dumps -type d -delete

test_installs:
	python -m pip install .[tests]
	python -m pip install git+https://github.com/huggingface/transformers.git
