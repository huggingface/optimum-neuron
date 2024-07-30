from transformers import AutoTokenizer
from optimum.neuron import NeuronModelForCausalLM

model = NeuronModelForCausalLM.from_pretrained("./mistral_neuron", local_files_only=True)

tokenizer = AutoTokenizer.from_pretrained("./mistral_neuron")
tokenizer.pad_token_id = tokenizer.eos_token_id

def format(message):
    message = f"[INST] {message} [/INST]\n\n"
    return message

def format_chat_prompt(message, history, max_tokens):
    chat = []

    for interaction in history:
        chat.append({"role": "user", "content": interaction[0]})
        chat.append({"role": "assistant", "content": interaction[1]})

    chat.append({'role': 'user', 'content': message})

    for i in range(0, len(chat), 2):
        prompt = tokenizer.apply_chat_template(chat[i:], tokenize=False)

        tokens = tokenizer(prompt)
        if len(tokens.input_ids) <= max_tokens:
            return prompt

    raise SystemError


def chat(history, max_tokens):
    message = input("Enter input: ")

    if message == "quit":
        return

    inputs = tokenizer(format_chat_prompt(message, history, max_tokens), return_tensors="pt")

    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.9,
        top_k=50,
        top_p=0.9
    )

    answer = tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)

    print(answer)

    history.append([message, answer])

    chat(history, max_tokens)

if __name__ == "__main__":
    history = []
    max_tokens=4096
    chat(history, max_tokens)
