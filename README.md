1. Setting up SSH Key:
```bash
ssh-keygen -t rsa -b 4096 -C "your_email"
eval "$(ssh-agent -s)"
ssh-add ~/.ssh/id_rsa
cat ~/.ssh/id_rsa.pub
ssh -T git@github.com
git config user.name "your_name"
git config user.email "your_email"
```

2. Installing miniforge:
```bash
cd ~
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
bash Miniforge3-Linux-x86_64.sh
vi ~/.bashrc
```

3. Adding environment variables to ~/.bashrc:
```bash
export MY_HOME=
export MY_PROJECT=MAS-TTS
export MY_OUTPUT=
export OPENAI_API_KEY=
export TOGETHER_API_KEY=
export TOGETHER_BASE_URL=
export DEEPINFRA_TOKEN=
export DEEPINFRA_BASE_URL=
export DEEPSEEK_BASE_URL=
export DEEPSEEK_API_KEY=
export WANDB_USER_NAME=
export WANDB_API_KEY=
export HUGGING_FACE_HUB_TOKEN=
export HUGGING_FACE_USER_NAME=
```
```bash
source ~/.bashrc
```

4. Installing requirements:
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
cd $MY_PROJECT
python model_download.py
```

5. download models:
```bash
cd $MY_PROJECT/LLaMA-Factory
python model_download.py
```

6. Config new models:
Add new models: gpt-4o-mini, o3-mini, deepseek-chat, deepseek-reasoner, Qwen, etc.
```
# local models
add in agentverse/llms/__init__.py


# remote models
add in agentverse/llms/openai.py
```

7. Config new datasets:
Add new datasets or tasks: AIME 2024, MATH-500, GPQA Diamond, etc.
```
# data
add data in data

# tasks
register tasks in dataloader/__init__.py

# configs
add configs in agentverse/tasks/tasksolving
```

