# Optimum Neuron documentation

1. Setup
```bash
pip install "git+https://github.com/huggingface/doc-builder.git" watchdog --upgrade
```

2. Local Development
```bash
doc-builder notebook-to-mdx notebooks/ec2/text-classification/fine_tune_bert.ipynb --output_dir docs/source/training_tutorials/
doc-builder notebook-to-mdx notebooks/sagemaker/deploy-llama-3-3-70b.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/sagemaker/deploy-mixtral-8x7b.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/sentence-transformers/sentence-transformers.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/text-generation/llama2-13b-chatbot.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/stable-diffusion/stable-diffusion-txt2img.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/stable-diffusion/stable-diffusion-xl-txt2img.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/text-generation/CodeLlama-7B-Compilation.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ec2/feature-extraction/qwen_embedding.ipynb --output_dir docs/source/inference_tutorials/
doc-builder notebook-to-mdx notebooks/ie/compare-book-translations.ipynb --output_dir docs/source/inference_tutorials/
sed 's#](\./#](https://github.com/huggingface/optimum-neuron/blob/main/notebooks/#g' notebooks/README.md > docs/source/notebooks.mdx
doc-builder preview optimum.neuron docs/source/
```
3. Build Docs
```bash
doc-builder build optimum.neuron docs/source/ --build_dir build/
```

## Add assets/Images

Adding images/assets is only possible through `https://` links meaning you need to use `https://raw.githubusercontent.com/huggingface/optimum-neuron/main/docs/assets/` prefix.

example

```bash
<img src="https://raw.githubusercontent.com/huggingface/optimum-neuron/main/docs/assets/0_login.png" alt="Login" />
```
