The code is based on the Agentverse and LLaMA-Factory, LICENSE of these projects are kept in this repository.

1. Installing miniforge:
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
vi ~/.bashrc
```

2. Adding environment variables to ~/.bashrc:
```bash
# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
# <<< conda initialize <<<
export MY_HOME=""
export MY_PROJECT=""
export OPENAI_API_KEY=""
export TOGETHER_API_KEY=""
export TOGETHER_BASE_URL=""
export DEEPSEEK_BASE_URL=""
export DEEPSEEK_API_KEY=""
export WANDB_USER_NAME=""
export WANDB_API_KEY=""
export HUGGING_FACE_HUB_TOKEN=""
export HUGGING_FACE_USER_NAME=""
source ~/.bashrc
```

3. Installing requirements:
```bash
# Create and activate conda environment
conda create -n agent python=3.11
conda activate agent
cd $MY_PROJECT
pip install -e .
pip install transformers hf_transfer aiohttp tenacity vllm
pip install -U openai
python -m spacy download en_core_web_sm

conda create -n llamafactory python=3.11
conda activate llamafactory
cd $MY_PROJECT/LLaMA-Factory
pip install -e ".[torch,metrics]"
pip install vllm
pip install deepspeed flash-attn wandb
pip install unsloth==2025.2.14 unsloth_zoo==2025.2.7 
```

4. Train model:
The full dataset of M500 is in `data/M500.jsonl`.
```bash
bash run/sft.sh
```

5. Run benchmark:
```bash
bash run/benchmark.sh
```


