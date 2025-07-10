# 🚀 Fine-Tune Qwen3 on AWS Trainium

> **Supercharge your Qwen3 models with blazing-fast training on AWS Trainium accelerators**

This example demonstrates how to fine-tune the powerful Qwen3 model using `optimum-neuron` for maximum performance on AWS Trainium chips.

✨ **Key Features:**
- 🎯 **Optimized for Trainium**: Native support for AWS's most advanced AI accelerators
- 🔧 **Plug-and-play**: Easy adaptation to other datasets and supported models
- 🎨 **Flexible**: Customize training parameters to fit your specific needs

## 🛠️ Quick Start

### Step 1: Compilation (Optional)

For first-time training or custom configurations, compile your model:

```bash
neuron_parallel_compile ./finetune_qwen3.sh
```

> 💡 **Pro Tip**: Skip this step by using pre-compiled models from our [HuggingFace Hub cache](https://huggingface.co/aws-neuron/optimum-neuron-cache) for instant training!

### Step 2: Launch Training

Start fine-tuning with a single command:

```bash
./finetune_qwen3.sh
```

## 📊 Performance Results

### `Qwen/Qwen3-8B` Benchmark



