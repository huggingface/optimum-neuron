from transformers import AutoTokenizer
from optimum.neuron.trainers.grpo_trainer import NeuronGRPOTrainer

model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

trainer = NeuronGRPOTrainer(
    model=model_name,
    tokenizer=tokenizer,
    args=None,
    reward_funcs=[lambda prompts, completions: [1.0]*len(completions)],
)

print("Trainer instantiated successfully!")
