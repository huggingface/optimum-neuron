# 🚀 Fine-Tune BERT on AWS Trainium

> **Transform text classification with lightning-fast BERT training on AWS Trainium accelerators**

This example demonstrates how to fine-tune BERT for emotion classification using `optimum-neuron` on the [emotion dataset](https://huggingface.co/datasets/dair-ai/emotion).

✨ **Key Features:**
- 🎯 **Cost-effective**: Only ~$0.18 for complete training on `trn1.2xlarge`
- ⚡ **Fast training**: 7.5 minutes for 3 epochs with distributed training
- 🔧 **Ready-to-use**: Emotion classification with 6 emotion categories

## 🛠️ Quick Start

### Step 1: Compilation (Optional)

For first-time training or custom configurations, compile your model:

```bash
neuron_parallel_compile ./finetune_bert.sh
```

> 💡 **Pro Tip**: Skip this step by using pre-compiled models from our [HuggingFace Hub cache](https://huggingface.co/aws-neuron/optimum-neuron-cache) for instant training!

### Step 2: Launch Training

Start fine-tuning with a single command:

```bash
./finetune_bert.sh
```

### Custom Training

```bash
python finetune_bert.py --model_id bert-base-uncased --epochs 3
```

---

📖 **Need more details?** Check out the [complete tutorial](https://huggingface.co/docs/optimum-neuron/training_tutorials/finetune_bert) for step-by-step guidance.
