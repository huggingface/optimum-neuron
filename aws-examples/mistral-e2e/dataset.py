from datasets import DatasetDict, load_dataset

def format(sample):
    sample['text'] = f"<s>[INST] {sample['question']} [/INST]\n\n{sample['answer']}</s>"
    return sample

# Downloads the gsm8k dataset directly from Hugging Face.
dataset = load_dataset("gsm8k", "main")

# We need to split the dataset into a training, and validation set.
# Note gsm8k has 'test', we rename to 'validation' for our training script.
train = dataset['train']
validation = dataset['test']

# Map the format function on all elements of the training and validation splits.
# Also removes the question and answer columns we no longer need.
train = train.map(format, remove_columns=list(train.features))
validation = validation.map(format, remove_columns=list(validation.features))

# Create a new DatasetDict with our train and validation splits.
dataset = DatasetDict({"train": train, "validation": validation})

dataset.save_to_disk('dataset_formatted')
