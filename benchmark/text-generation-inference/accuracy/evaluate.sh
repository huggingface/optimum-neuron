model_id=${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}
tasks=${3:-gsm8k}
batch_size=${2:-1}

export HF_TOKEN=$(cat ~/.cache/huggingface/token)

base_url="http://127.0.0.1:8080/v1/chat/completions"

lm_eval --model local-chat-completions \
        --tasks ${tasks} \
        --model_args model=${model_id},base_url=${base_url},num_concurrent=${batch_size},max_retries=3,tokenized_requests=False \
        --apply_chat_template
