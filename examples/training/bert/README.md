# 🚀 Fine-Tune BERT on AWS Trainium

> **Transform text classification with lightning-fast BERT training on AWS Trainium accelerators**

This example demonstrates how to fine-tune BERT for emotion classification using `optimum-neuron` on the [emotion dataset](https://huggingface.co/datasets/dair-ai/emotion).

✨ **Key Features:**
- 🎯 **Cost-effective**: Only ~$0.18 for complete training on `trn1.2xlarge`
- ⚡ **Fast training**: 7.5 minutes for 3 epochs with distributed training
- 🔧 **Ready-to-use**: Emotion classification with 6 emotion categories

## 🛠️ Quick Start

### Launch Training

Start fine-tuning with a single command:

```bash
./fine_tune_bert.sh
```

### Custom Training

```bash
python fine_tune_bert.py --model_id bert-base-uncased --epochs 3
```

---

📖 **Need more details?** Check out the [complete tutorial](https://huggingface.co/docs/optimum-neuron/training_tutorials/fine_tune_bert) for step-by-step guidance.