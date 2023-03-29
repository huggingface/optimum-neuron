# Optimum Neuron documentation

1. Setup
```bash
pip install hf-doc-builder==0.4.0 watchdog --upgrade
```

2. Local Development
```bash
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

