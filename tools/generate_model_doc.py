# coding=utf-8
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tool to generate model documentation for Neuron models."""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple

import requests
import yaml

from optimum.exporters.tasks import TasksManager


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

LICENSE_HEADER = '''<!---
Copyright 2025 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->'''

EXPORT_TEMPLATE = '''## Export to Neuron

To deploy 🤗 [Transformers](https://huggingface.co/docs/transformers/index) models on Neuron devices, you first need to compile the models and export them to a serialized format for inference. Below are two approaches to compile the model, you can choose the one that best suits your needs. Here we take the `{task}` as an example:

### Option 1: CLI
  
You can export the model using the Optimum command-line interface as follows:

```bash
optimum-cli export neuron --model {model_id} --task {task} --batch_size 1{cli_args} {output_dir}/
```

> [!TIP]
> Execute `optimum-cli export neuron --help` to display all command line options and their description.

### Option 2: Python API

```python
from optimum.neuron import {model_class}

input_shapes = {{"batch_size": 1{extra_shapes}}}
compiler_args = {{"auto_cast": "matmul", "auto_cast_type": "bf16"}}
neuron_model = {model_class}.from_pretrained(
    "{model_id}",
    export=True,
    **input_shapes,
    **compiler_args,
)

# Save locally
neuron_model.save_pretrained("{output_dir}")

# Upload to the HuggingFace Hub
neuron_model.push_to_hub(
    "{output_dir}", repository_id="my-neuron-repo"  # Replace with your HF Hub repo id
)
```'''

def clean_overview_text(text: str) -> str:
    """Clean extracted overview but keep links and paragraph breaks."""
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'[ \t]+$', '', text, flags=re.MULTILINE)
    return text.strip()

def get_model_title_from_docs(content: str, model_name: str) -> str:
    """Extract the model title as written in transformers docs."""
    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

    content = content.strip()

    # Try to find the first level-1 heading (single #)
    heading_match = re.search(r'^\s*#\s+([^#\n]+)', content, flags=re.MULTILINE)

    if heading_match:
        title = heading_match.group(1).strip()
        if title and not title.isspace():
            return title

    simple_heading_match = re.search(r'#\s+([^\n]+)', content)
    if simple_heading_match:
        title = simple_heading_match.group(1).strip()
        if title and not title.isspace():
            return title

    # Fallback to uppercase model name if no title found
    return model_name.upper()

def extract_overview_from_content(content: str, model_name: str) -> str:
    """Extract only the overview section from transformers doc content."""

    content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)
    content = re.sub(r'/\*.*?\*/', '', content, flags=re.DOTALL)
    content = re.sub(r'^---\s*\n.*?\n---\s*\n', '', content,
                     flags=re.DOTALL | re.MULTILINE)

    # Build a stop marker for overview extraction
    stop_pattern = r'(?=\n#{1,3}\s|\n>\s*\[!TIP\]|\n>\s*\[!NOTE\]|\n>\s*\[!IMPORTANT\]|\Z)'

    patterns = [
        rf'## Overview\s*\n(.*?){stop_pattern}',
        rf'# Overview\s*\n(.*?){stop_pattern}',
        rf'## Model description\s*\n(.*?){stop_pattern}',
        rf'# Model description\s*\n(.*?){stop_pattern}',
        rf'## Introduction\s*\n(.*?){stop_pattern}',
        rf'## Summary\s*\n(.*?){stop_pattern}',
        # Sometimes the overview is right after the model name
        rf'#\s*{re.escape(model_name)}\s*\n(.*?){stop_pattern}',
    ]

    for pattern in patterns:
        overview_match = re.search(pattern, content,
                                   re.DOTALL | re.MULTILINE | re.IGNORECASE)
        if overview_match:
            overview_text = clean_overview_text(overview_match.group(1))
            if overview_text and len(overview_text) > 50:
                logger.info(f"Found overview for {model_name} using pattern: {pattern}")
                return overview_text

    return ""

def generate_fallback_overview(model_name: str) -> str:
    """Generate a fallback overview when transformers documentation is not available"""
    # fallback
    return f"The {model_name.upper()} model is a transformer-based model. Please refer to the original paper and Transformers documentation for detailed information about the model architecture and training procedure."

def get_transformers_overview(model_name: str, local_transformers_path: Optional[Path] = None) -> Tuple[str, str]:
    """Fetch model overview from transformers documentation (local or remote)
    Returns: (overview_text, model_title)
    """
    logger.info(f"Fetching overview for {model_name}")

    # Try local transformers documentation first if path is provided
    if local_transformers_path and local_transformers_path.exists():
        local_doc_paths = [
            local_transformers_path / "docs" / "source" / "en" / "model_doc" / f"{model_name}.md",
            local_transformers_path / "docs" / "source" / "en" / "model_doc" / f"{model_name}.mdx",
        ]

        for local_path in local_doc_paths:
            if local_path.exists():
                try:
                    logger.debug(f"Reading local file: {local_path}")
                    content = local_path.read_text(encoding='utf-8')
                    overview = extract_overview_from_content(content, model_name)
                    model_title = get_model_title_from_docs(content, model_name)
                    if overview:
                        logger.info(f"Successfully got overview from local file: {local_path}")
                        return overview, model_title
                except Exception as e:
                    logger.debug(f"Failed to read local file {local_path}: {e}")
                    continue

    # Fallback to remote URLs if local files not found
    urls_to_try = [
        f"https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/{model_name}.md",
        f"https://raw.githubusercontent.com/huggingface/transformers/main/docs/source/en/model_doc/{model_name}.mdx",
    ]

    for url in urls_to_try:
        try:
            logger.debug(f"Trying URL: {url}")
            response = requests.get(url, timeout=30)
            if response.status_code == 200:
                content = response.text
                logger.debug(f"Successfully fetched content from {url}")
                overview = extract_overview_from_content(content, model_name)
                model_title = get_model_title_from_docs(content, model_name)
                if overview:
                    logger.info(f"Successfully got overview from remote URL: {url}")
                    return overview, model_title

        except Exception as e:
            logger.debug(f"Failed to fetch from {url}: {e}")
            continue

    logger.warning(f"Could not fetch overview for {model_name}")
    return generate_fallback_overview(model_name), model_name.upper()


def infer_task_for_model(model_id: str) -> str:
    """Infer task for a specific model using TasksManager."""
    logger.info(f"Infer task for model {model_id}")
    try:
        task = TasksManager.infer_task_from_model(model_id)
    except Exception as e:
        logger.error(f"Failed to infer task for model {model_id}: {e}")
        task = "unknown"
    return task

def get_model_task(model_dir: Path, model_id: str) -> str:
    """Get supported task by using the model type and TasksManager"""
    logger.info(f"Getting task for model {model_id} in {model_dir}")

    # Infer task using TasksManager
    task = infer_task_for_model(model_id)

    if task == "unknown":
        logger.info("No task found through TasksManager, trying to infer from class names")
        task = "feature-extraction"  # Default fallback task


    logger.info(f"Final task for {model_dir.name}: {task}")
    return task


def get_model_classes(model_dir: Path) -> List[str]:
    """Get all Python class names from the model directory, keeping appearance order."""
    logger.info(f"Getting model classes from {model_dir}")
    classes = []
    seen = set()

    for file in model_dir.glob("*.py"):
        if file.name == "__init__.py":
            continue

        try:
            with open(file, 'r', encoding='utf-8') as f:
                content = f.read()
                # Match any Python class name
                class_matches = re.findall(r'class\s+([A-Za-z_]\w*)(?:\(|:)', content)
                for cls in class_matches:
                    if cls not in seen:
                        seen.add(cls)
                        classes.append(cls)
                logger.debug(f"Found classes in {file.name}: {class_matches}")

        except Exception as e:
            logger.error(f"Error reading {file}: {e}")

    logger.info(f"Model classes for {model_dir.name}: {classes}")
    return classes

def get_task_model_class(task: str) -> str:
    """Get the appropriate NeuronModel class for a task"""
    task_to_class = {
        "feature-extraction": "NeuronModelForFeatureExtraction",
        "text-classification": "NeuronModelForSequenceClassification",
        "token-classification": "NeuronModelForTokenClassification",
        "question-answering": "NeuronModelForQuestionAnswering",
        "multiple-choice": "NeuronModelForMultipleChoice",
        "fill-mask": "NeuronModelForMaskedLM",
        "object-detection": "NeuronModelForObjectDetection",
        "image-classification": "NeuronModelForImageClassification",
        "semantic-segmentation": "NeuronModelForSemanticSegmentation",
        "automatic-speech-recognition": "NeuronModelForCTC",
        "audio-classification": "NeuronModelForAudioClassification",
        "text-generation": "NeuronModelForCausalLM",
        "text2text-generation": "NeuronModelForSeq2SeqLM",
    }
    return task_to_class.get(task, "NeuronModel")

def get_task_extra_shapes(task: str) -> str:
    """Get additional input shapes needed for the task"""
    task_to_shapes = {
        "text-classification": ', "sequence_length": 128',
        "token-classification": ', "sequence_length": 128',
        "question-answering": ', "sequence_length": 384',
        "multiple-choice": ', "sequence_length": 128, "num_choices": 4',
        "fill-mask": ', "sequence_length": 128',
        "feature-extraction": ', "sequence_length": 128',
        "object-detection": ', "height": 800, "width": 1333',
        "image-classification": ', "height": 224, "width": 224',
        "semantic-segmentation": ', "height": 224, "width": 224',
        "automatic-speech-recognition": ', "audio_sequence_length": 100000',
        "audio-classification": ', "audio_sequence_length": 100000',
        "text-generation": ', "sequence_length": 128',
        "text2text-generation": ', "sequence_length": 128',
    }
    return task_to_shapes.get(task, "")

def get_task_cli_args(task: str) -> str:
    """Get CLI arguments needed for the task"""
    task_to_args = {
        "text-classification": " --sequence_length 128",
        "token-classification": " --sequence_length 128",
        "question-answering": " --sequence_length 384",
        "multiple-choice": " --sequence_length 128 --num_choices 4",
        "fill-mask": " --sequence_length 128",
        "feature-extraction": " --sequence_length 128",
        "object-detection": " --height 800 --width 1333",
        "image-classification": " --height 224 --width 224",
        "semantic-segmentation": " --height 224 --width 224",
        "automatic-speech-recognition": " --audio_sequence_length 100000",
        "audio-classification": " --audio_sequence_length 100000",
        "text-generation": " --sequence_length 128",
        "text2text-generation": " --sequence_length 128",
    }
    return task_to_args.get(task, "")

def generate_model_doc(model_name: str, model_id: str, inference_dir: Path, docs_dir: Path, local_transformers_path: Optional[Path] = None):
    """Generate .mdx documentation for a model"""
    logger.info(f"Generating documentation for {model_name}")

    model_dir = inference_dir / model_name
    if not model_dir.is_dir():
        logger.error(f"Model directory {model_dir} does not exist")
        return None

    task = get_model_task(model_dir, model_id)

    overview, model_title = get_transformers_overview(model_name, local_transformers_path)
    print(f"Overview for {model_name}:\n{overview}\n")

    logger.info(f"Using primary task '{task}' for {model_name} documentation")

    model_class = get_task_model_class(task)
    extra_shapes = get_task_extra_shapes(task)
    cli_args = get_task_cli_args(task)

    doc_content = [
        LICENSE_HEADER,
        f"# {model_title}",
        "",
        "## Overview",
        "",
        overview,
        "",
        EXPORT_TEMPLATE.format(
            task=task,
            model_id=model_id,
            model_class=model_class,
            output_dir=f"{model_name}_{task.replace('-', '_')}_neuronx",
            extra_shapes=extra_shapes,
            cli_args=cli_args
        )
    ]

    # Add model classes using autodoc
    classes = get_model_classes(model_dir)
    if classes:
        for class_name in classes:
            doc_content.extend([
                "",
                f"## {class_name}",
                "",
                f"[[autodoc]] models.inference.{model_name}.{class_name}"
            ])
    else:
        logger.warning(f"No model classes found for {model_name}")

    # Ensure output directory exists
    output_dir = docs_dir / "model_doc" / "transformers"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{model_name}.mdx"
    try:
        output_file.write_text("\n".join(doc_content), encoding='utf-8')
        logger.info(f"Successfully created documentation: {output_file}")
    except Exception as e:
        logger.error(f"Failed to write documentation file {output_file}: {e}")
        return None
    return model_name, task, model_title

def update_toctree(model_name: str, model_category: str, docs_dir: Path, model_title: str):
    """Update _toctree.yml to include new model documentation"""
    logger.info(f"Updating _toctree.yml for {model_name}")

    toctree_file = docs_dir / "_toctree.yml"
    if not toctree_file.exists():
        logger.error(f"_toctree.yml file not found at {toctree_file}")
        return

    try:
        with open(toctree_file, 'r', encoding='utf-8') as f:
            toc = yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Failed to load _toctree.yml: {e}")
        return

    # Find the Models and Pipelines Inference API section
    for section in toc[0]["sections"]:
        if section.get("title") == "Models and Pipelines Inference API":
            transformers_section = None
            for subsection in section["sections"]:
                if subsection.get("title") == "Transformers Models":
                    transformers_section = subsection
                    break

            if not transformers_section:
                transformers_section = {
                    "title": "Transformers Models",
                    "sections": []
                }
                for category in ["TEXT MODELS", "VISION MODELS", "AUDIO MODELS", "MULTIMODAL MODELS"]:
                    category_section = {
                        "title": category,
                        "sections": []
                    }
                    transformers_section["sections"].append({
                        "title": category,
                        "sections": [],
                        "isExpanded": False
                    })
                section["sections"].insert(0, transformers_section)

            model_entry = {
                "local": f"model_doc/transformers/{model_name}",
                "title": model_title
            }

            category_titles = {
                "TEXT": "TEXT MODELS",
                "VISION": "VISION MODELS",
                "AUDIO": "AUDIO MODELS",
                "MULTIMODAL": "MULTIMODAL MODELS"
            }

            target_title = category_titles[model_category]

            for category_section in transformers_section["sections"]:
                if category_section["title"] == target_title:
                    exists = False
                    for entry in category_section["sections"]:
                        if entry.get("local") == model_entry["local"]:
                            exists = True
                            logger.info(f"Model {model_name} already exists in {model_category} section")
                            break

                    if not exists:
                        category_section["sections"].append(model_entry)
                        logger.info(f"Added {model_name} to {model_category} section")

                    category_section["sections"].sort(key=lambda x: x["title"])
                    break

            break

    try:
        with open(toctree_file, 'w', encoding='utf-8') as f:
            yaml.dump(toc, f, allow_unicode=True, sort_keys=False, default_flow_style=False)
        logger.info("Successfully updated _toctree.yml")
    except Exception as e:
        logger.error(f"Failed to write _toctree.yml: {e}")

def validate_model_directory(model_dir: Path) -> bool:
    """Validate that a model directory contains valid Python files"""
    if not model_dir.is_dir():
        return False

    python_files = [f for f in model_dir.glob("*.py") if f.name != "__init__.py"]

    if not python_files:
        logger.debug(f"No Python files found in {model_dir}")
        return False

    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
                if re.search(r'class\s+Neuron\w+', content):
                    return True
        except Exception as e:
            logger.debug(f"Error reading {py_file}: {e}")
            continue

    logger.debug(f"No Neuron model classes found in {model_dir}")
    return False

def main():
    """Main function to generate documentation for a single model"""
    parser = argparse.ArgumentParser(description='Generate documentation for a specific Neuron model')
    parser.add_argument('--model_name', required=True, help='Name of the model directory')
    parser.add_argument('--model_id', required=True, help='Model ID to use in examples (e.g., google-bert/bert-base-uncased)')
    parser.add_argument('--model_category', required=True, choices=['TEXT', 'VISION', 'AUDIO', 'MULTIMODAL'], help='Model category for documentation organization (e.g., TEXT, VISION)')
    parser.add_argument('--local_transformers', help='Path to local transformers repository')
    parser.add_argument('--base_dir', help='Base directory of optimum-neuron (default: parent of script directory)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    # Set base directory if provided, otherwise use parent of script directory
    base_dir = Path(args.base_dir) if args.base_dir else Path(__file__).parent.parent

    # Set paths
    inference_dir = base_dir / "optimum" / "neuron" / "models" / "inference"
    docs_dir = base_dir / "docs" / "source"

    # Validate model directory
    model_dir = inference_dir / args.model_name
    if not validate_model_directory(model_dir):
        logger.error(f"Invalid model directory: {model_dir}")
        return

    # Generate documentation
    result = generate_model_doc(
        model_name=args.model_name,
        model_id=args.model_id,
        inference_dir=inference_dir,
        docs_dir=docs_dir,
        local_transformers_path=Path(args.local_transformers) if args.local_transformers else None
    )

    if result:
        model_name, task, model_title = result
        # Update _toctree.yml with the new model documentation
        update_toctree(model_name, args.model_category, docs_dir, model_title)
    else:
        logger.error(f"Failed to generate documentation for {args.model_name}")

if __name__ == "__main__":
    main()
