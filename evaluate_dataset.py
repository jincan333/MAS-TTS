import json
import argparse
from openai import AsyncOpenAI
import os
import time
import asyncio
import re
import json
import spacy
from tqdm import tqdm
import multiprocessing
import subprocess
import unicodedata
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk.corpus import wordnet
from nltk.corpus import wordnet as wn
import inflect
import nltk
from typing import cast
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()
p = inflect.engine()

nlp = spacy.load("en_core_web_sm")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

math_datasets = ['aime2024', 'math500', 'gpqa_diamond']
code_datasets = ['humaneval', 'mbpp_sanitized', 'mbpp']
commongen_datasets = ['commongen']
tool_datasets = []
other_datasets = []

IRREGULAR_VERBS = {
    'lie': ['lay', 'lain', 'lying'],
    'lay': ['lie', 'laid', 'laying'],
    'wear': ['wore', 'worn'],
    'rise': ['rose', 'risen'],
    'fall': ['fell', 'fallen'],
    'have': ['had', 'having'],
    'do': ['did', 'doing'],
    'go': ['went', 'gone'],
    'see': ['saw', 'seen'],
    'read': ['read', 'read'],
    'take': ['took', 'taken'],
    'eat': ['ate', 'eaten'],
    'be': ['was', 'were', 'been'],
}

compare_prompt = r"""As an experienced teacher, evaluate if the student's final numerical answer matches the standard answer. Consider:

Correct (1) if:
- Final values are numerically equivalent (42 vs 42.0 vs 42.00)
- Different but equivalent representations (½ vs 0.5)
- Reasonable rounding differences (3.14 vs 3.1416)

Incorrect (0) if:
- Final values differ mathematically
- No answer provided

Ignore formatting and units. Only compare final numerical values.

Solution:
{solution}

Standard Answer:
{standard_answer}

Strictly respond with:
### Correctness: (1 if correct, 0 otherwise)"""

def invert(model):
    if model in ('deepseek-chat', 'deepseek-reasoner'):
        model = "deepseek-ai/DeepSeek-V3" if model == 'deepseek-chat' else "deepseek-ai/DeepSeek-R1"
    return model

def revert(model):
    if model in ('deepseek-ai/DeepSeek-V3', 'deepseek-ai/DeepSeek-R1'):
        model = "deepseek-chat" if model == "deepseek-ai/DeepSeek-V3" else "deepseek-reasoner"
    return model

def get_prompt(dataset):
    if dataset in math_datasets or dataset in other_datasets:
        return r"""As an experienced math teacher, carefully evaluate whether the given solution matches the standard answer. Follow these criteria:
1. The solution must contain a final answer
2. All answers in the solution must be checked against the standard answer
3. The reasoning leading to the answer must be mathematically sound
4. Only consider numerical equivalence (ignore formatting differences like 0.5 vs 1/2 if numerically equal)

Evaluation Process:
1. If multiple boxed answers exist, check all → Correct if ANY matches standard answer
2. If answer format differs but values are numerically equivalent → Consider correct
3. If solution contains correct answer but flawed reasoning → Consider incorrect

### Problem:
{problem}

### Solution:
{solution}

### Standard Answer:
{standard_answer}

Analyze the solution and determine if it satisfies all requirements for correctness. Respond your final judgement at the end of your response in the following format:
### Correctness: (0 or 1, 0=incorrect, 1=correct)"""
    elif dataset in code_datasets:
        return "Please solve the following code problem: "
    elif dataset in commongen_datasets:
        return "Please generate a response for the following prompt: "
    elif dataset in tool_datasets:
        return "Please solve the following problem: "

def parse_response(response, dataset):
    if dataset in math_datasets or dataset in other_datasets:
        match = re.search(r'### Correctness:\D*(\d+)', response)
        return int(match.group(1)) if match else 0
    elif dataset in code_datasets:
        return response
    elif dataset in commongen_datasets:
        return response
    elif dataset in tool_datasets:
        return response
    

def extract_answer(response):
    response = response.replace('\x08', r'\b')
    # Find the position of \boxed{
    start_idx = response.rfind(r'\boxed{')
    if start_idx == -1:
        return response
    
    # Start after \boxed{
    start_idx += 7
    # Track nested braces
    brace_count = 1
    end_idx = start_idx
    
    # Process until matching closing brace is found
    while end_idx < len(response) and brace_count > 0:
        if response[end_idx] == '{':
            brace_count += 1
        elif response[end_idx] == '}':
            brace_count -= 1
        end_idx += 1
    
    # If we found a matching closing brace
    if brace_count == 0:
        return response[start_idx:end_idx-1]
    else:
        return response

def get_spend(model, send_tokens, recv_tokens) -> int:
        input_cost_map = {
            "gpt-3.5-turbo": 0.0015,
            "gpt-3.5-turbo-16k": 0.003,
            "gpt-3.5-turbo-0613": 0.0015,
            "gpt-3.5-turbo-16k-0613": 0.003,
            "gpt-3.5-turbo-1106": 0.0005,
            "gpt-3.5-turbo-0125": 0.0005,
            "gpt-4": 0.03,
            "gpt-4-0613": 0.03,
            "gpt-4-32k": 0.06,
            "gpt-4-1106-preview": 0.01,
            "gpt-4-0125-preview": 0.01,
            "llama-2-7b-chat-hf": 0.0,
            "gpt-4o": 0.0025,
            "gpt-4o-mini": 0.00015,
            "o3-mini": 0.0011,
            "deepseek-chat": 0.00125,
            "deepseek-reasoner": 0.003,
            "qwen2.5-32b-instruct": 0,
            "s1.1-32b": 0,
            "deepseek-r1-distill-qwen-32b": 0,
            "d1-32b": 0,
            "m1-32b": 0,
        }

        output_cost_map = {
            "gpt-3.5-turbo": 0.002,
            "gpt-3.5-turbo-16k": 0.004,
            "gpt-3.5-turbo-0613": 0.002,
            "gpt-3.5-turbo-16k-0613": 0.004,
            "gpt-3.5-turbo-1106": 0.0015,
            "gpt-3.5-turbo-0125": 0.0015,
            "gpt-4": 0.06,
            "gpt-4-0613": 0.06,
            "gpt-4-32k": 0.12,
            "gpt-4-1106-preview": 0.03,
            "gpt-4-0125-preview": 0.03,
            "llama-2-7b-chat-hf": 0.0,
            "gpt-4o": 0.010,
            "gpt-4o-mini": 0.0006,
            "o3-mini": 0.0044,
            "deepseek-chat": 0.00125,
            "deepseek-reasoner": 0.007,
            "qwen2.5-32b-instruct": 0,
            "s1.1-32b": 0,
            "deepseek-r1-distill-qwen-32b": 0,
            "d1-32b": 0,
            "m1-32b": 0,
        }

        if model not in input_cost_map or model not in output_cost_map:
            raise ValueError(f"Model type {model} not supported")

        return (
            send_tokens * input_cost_map[model] / 1000.0
            + recv_tokens * output_cost_map[model] / 1000.0
        )

async def agenerate_response(
        user_prompt: str = "",
        model: str = "deepseek-chat",
    ) -> dict:
        messages = [{"role": "user", "content": user_prompt}]
        if model in ('deepseek-chat', 'deepseek-reasoner'):
            api_key = os.environ.get("TOGETHER_API_KEY")
            base_url = os.environ.get("TOGETHER_BASE_URL")
        elif model in ["gpt-4o", "gpt-4o-mini", 'o3-mini']:
            api_key = os.environ.get("OPENAI_API_KEY")
            base_url = None

        async_openai_client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=3000,
        )
        for _ in range(3):
            try:
                model = invert(model)
                response = await async_openai_client.chat.completions.create(
                    messages=messages,
                    model=model,
                    max_tokens=8192,
                    temperature=0.7,
                    top_p=0.95,
                    n=1,
                    stop=None,
                    presence_penalty=0,
                    frequency_penalty=0,
                )
                await async_openai_client.close()
                model = revert(model)
                content = response.choices[0].message.content
                messages = messages + [{"role": "assistant", "content": content}]
                spend = get_spend(model, response.usage.prompt_tokens, response.usage.completion_tokens)
                return content, messages, spend
            except Exception as e:
                await async_openai_client.close()
                model = revert(model)
                print(f"Unexpected error: {type(e).__name__}: {str(e)}")
                time.sleep(1)
                print("retrying...")
                continue
        return 'I cannot solve the problem.', messages, 0

def get_solution(result, dataset):
    if 'response' in result and result['response'] != '':
        if dataset in math_datasets or dataset in other_datasets:
            if r"\boxed{" in result['response']:
                return extract_answer(result['response'])
            else:
                return result['response']
        elif dataset in commongen_datasets:
            if extract_answer(result['response']) not in ['I cannot solve the problem.', 'paragraph']:
                return extract_answer(result['response'])
            else:
                return result['response']
        elif dataset in code_datasets:
            return result['response']
        
    if 'logs' in result and len(result['logs']) > 0:
        reversed_logs = reversed(result['logs'])
        for item in reversed_logs:
            if 'module' in item and item['module'] == 'Decision Maker' and item['content'] != '':
                if dataset in math_datasets:
                    if extract_answer(item['content']) != 'I cannot solve the problem.':
                        return extract_answer(item['content'])
                elif dataset in commongen_datasets:
                    if extract_answer(item['content']) != 'I cannot solve the problem.':
                        return extract_answer(item['content'])
                    else:
                        return item['content']
                elif dataset in code_datasets:
                    return item['content']
        return 'I cannot solve the problem.'
    else:
        return 'I cannot solve the problem.'
    
def execute_command(command: str, result_list) -> str:
    result = subprocess.run(command, capture_output=True, shell=True, encoding="utf-8")
    result_list.append(f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}")


def get_coverage_score(preds, concept_sets):
    preds = [p for p in preds]
    covs = []
    missings = []
    for p, cs in zip(preds, concept_sets):
        cs = set(cs)
        lemmas = set()
        for token in nlp(p):
            lemmas.add(token.lemma_)
        cov = len(lemmas & cs) / len(cs)
        covs.append(cov)
        missings.append(cs - lemmas)
    return sum(covs) / len(covs), missings

def normalize_text(text):
    """Transform text into readable form while preserving content"""
    # Convert Unicode escape sequences to characters
    text = re.sub(r'\\u([a-fA-F0-9]{4})', lambda m: chr(int(m.group(1), 16)), text)
    # Normalize Unicode characters (e.g. accented characters)
    text = unicodedata.normalize('NFKC', text)
    # Remove special symbols but preserve all letters and numbers
    text = re.sub(r'[^\w\s-]', ' ', text)  # \w includes all alphanumeric characters
    
    return text

def get_wordnet_pos(treebank_tag):
    """Map POS tag to first character used by WordNetLemmatizer"""
    tag = treebank_tag[0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_tokens(tokens):
    """Lemmatize words with POS tagging"""
    tagged = pos_tag(tokens)
    return [lemmatizer.lemmatize(word.lower(), pos=get_wordnet_pos(tag)) 
            for word, tag in tagged]

def morphological_analyze(word):
    """Generate all possible morphological variants"""
    variants = set()
    clean_word = re.sub(r"[^a-zA-Z]", "", word).lower()
    clean_word_str = str(clean_word)
    singular_result = p.singular_noun(cast(str, clean_word_str))
    if singular_result: # singular_noun returns False if already singular or not recognized
        singular = str(singular_result) # Ensure result is string
        variants.add(singular)
        variants.add(stemmer.stem(singular))
    else:
        plural_result = p.plural(cast(str, clean_word_str))
        if plural_result and plural_result != clean_word_str: # plural returns the plural form string
            plural = str(plural_result) # Ensure result is string
            variants.add(plural)
            variants.add(stemmer.stem(plural))

    # Base forms
    variants.add(clean_word_str)
    variants.add(stemmer.stem(clean_word_str))

    # Enhanced verb handling for irregular forms
    for pos in [wn.VERB, wn.NOUN, wn.ADJ]:  # Check multiple POS tags
        lemma = lemmatizer.lemmatize(clean_word_str, pos=pos)
        if lemma != clean_word_str:
            variants.update([lemma, stemmer.stem(lemma)])

    # Special handling for past participles
    if clean_word_str.endswith(('ed', 'en')):
        verb_lemma = lemmatizer.lemmatize(clean_word_str, pos=wn.VERB)
        variants.update([verb_lemma, stemmer.stem(verb_lemma)])

    for base, forms in IRREGULAR_VERBS.items():
        if clean_word_str == base or clean_word_str in forms:
            variants.add(base)
            variants.update(forms)
            variants.add(stemmer.stem(base))
            variants.update(stemmer.stem(f) for f in forms)

    return variants


def count_matches(problem_words, solution_text):
    # Normalize and clean text
    solution_text = normalize_text(solution_text)
    solution_text = re.sub(r"[^a-zA-Z\s-]", " ", solution_text)
    
    # Process solution text with enhanced hyphen handling
    solution_tokens = word_tokenize(solution_text)
    solution_lemmas = set()
    
    for token in solution_tokens:
        # Process whole token and hyphen-split parts
        parts = token.split('-')
        for part in parts:
            solution_lemmas.update(morphological_analyze(part))
        solution_lemmas.update(morphological_analyze(token))  # Handle compound words
    
    # Process problem words
    problem_lemmas = set()
    for word in problem_words:
        # Split hyphenated problem words too
        for part in word.split('-'):
            problem_lemmas.update(morphological_analyze(part))
    
    # Find matches using comprehensive lemma sets
    missing_words = [
        word for word in problem_words
        if not any(morphological_analyze(part) & solution_lemmas 
                 for part in word.split('-'))
    ]
    
    coverage = 1 - len(missing_words)/len(problem_words) if problem_words else 0
    return coverage, missing_words

def main():
    args = argparse.ArgumentParser()
    args.add_argument('--judge_model', type=str, default='')
    args.add_argument('--model', type=str, default='d1-32b')
    args.add_argument('--dataset', type=str, default='commongen')
    args.add_argument('--output_path', type=str, default=None)
    args = args.parse_args()

    model = args.model
    dataset = args.dataset
    data_path = f'data/{dataset}/test.jsonl'
    if args.output_path is None:
        result_path = f'results/tasksolving/{dataset}/{model}/results.jsonl'
    else:
        result_path = f'results/tasksolving/{dataset}/{model}/{args.output_path}/results.jsonl'

    data = []
    with open(data_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))

    result = []
    with open(result_path, 'r') as f:
        for line in f:
            result.append(json.loads(line))

    prompt = get_prompt(dataset)
    self_judge_correct = 0
    correct = 0
    spend = 0
    total = len(data)
    if args.output_path is None:
        evaluation_path = f'results/tasksolving/{dataset}/{model}/evaluation.json'
    else:
        evaluation_path = f'results/tasksolving/{dataset}/{model}/{args.output_path}/evaluation.json'
    all_messages = []
    for i in range(len(result)):
        if result[i]['logs'][-1]['content'] == "Good score! Accept!":
            self_judge_correct += 1
        agentverse_spend = result[i]['spend'] if 'spend' in result[i] else 0
        spend += agentverse_spend
        problem = result[i]['input']
        solution = get_solution(result[i], dataset)
        standard_answer = result[i]['label']
        if dataset in math_datasets:
            if args.judge_model:
                user_prompt = prompt.replace(r'\boxed{answer}', r'\boxed*answer*').format(problem=problem, solution=solution, standard_answer=standard_answer)
                user_prompt = user_prompt.replace(r'\boxed*answer*', r'\boxed{answer}')
                response, messages, eval_spend = asyncio.run(agenerate_response(user_prompt, args.judge_model))
            else:
                if dataset == "gpqa_diamond" and (("B)" in solution and "A)" not in solution and "C)" not in solution and "D)" not in solution) or solution.strip() == "B"):
                    response = "### Correctness: 1"
                    messages = None
                    eval_spend = 0
                else:
                    user_prompt = compare_prompt.format(solution=solution, standard_answer=standard_answer)
                    response, messages, eval_spend = asyncio.run(agenerate_response(user_prompt, "deepseek-chat"))
            current_correct = parse_response(response, dataset)
            correct += current_correct
        elif dataset in commongen_datasets:
            current_correct, missing_words = count_matches(result[i]['input'], solution)
            messages = [{"problem": problem, "solution": solution, "missing words": ", ".join(map(str, missing_words))}]
            correct += current_correct
            eval_spend = 0
        elif dataset in code_datasets:
            test_cases = standard_answer
            function_names = re.findall(r'^def\s+([a-zA-Z0-9_]+)\s*\(', solution, re.MULTILINE)
            if function_names:
                solution_function_name = function_names[-1]  # Get the last function name
            else:
                print(solution)
            if dataset == "humaneval":
                test_code = solution + "\n\n" + test_cases + f"\n\n# Run tests\nif __name__ == '__main__':\n    check({solution_function_name})\n"
                test_code += "\n    print('<All tests passed>')"
            elif dataset in ["mbpp_sanitized", "mbpp"]:
                test_code = "import math\n\n" + solution + "\n\n"
                test_code += "\n".join(test_cases)
                test_code += "\nprint('<All tests passed>')"
                
            code_path = f"results/tasksolving/{dataset}/{model}/test.py"
            with open(code_path, 'w') as f:
                f.write(test_code)
                f.flush()
            manager = multiprocessing.Manager()
            result_list = manager.list()
            p = multiprocessing.Process(target=execute_command, args=(f"python {code_path}", result_list))
            p.start()
            p.join(timeout=10)
            if p.is_alive():
                p.terminate()
                p.join()
                response = "Timeout"
            else:
                response = result_list[0] if result_list else "No output"
            if "<All tests passed>" in response:
                current_correct = 1
            else:
                current_correct = 0
            messages = [{"problem": problem, "solution": solution, "test_cases": test_cases, "test_code": test_code, "test_result": response}]
            correct += current_correct
            eval_spend = 0
        spend += eval_spend
        current_message = {"messages": messages, "spend": agentverse_spend+eval_spend, "correct": current_correct}
        all_messages.append(current_message)
        with open(evaluation_path, 'a') as f:
            f.write(json.dumps(current_message, indent=4) + ',\n')

    evaluation = {
        'judge_model': args.judge_model,
        'model': model,
        'dataset': dataset,
        'spend': spend,
        'total': total,
        'evaluated': len(result),
        'evaluation_rate': f'{100 * len(result)/total:.2f}',
        'self_judge_correction_rate': f'{100 * self_judge_correct/total:.2f}',
        'actual_correction_rate': f'{100 * correct/total:.2f}'
    }
    print(json.dumps(evaluation, indent=4))

    all_messages.insert(0, evaluation)
    with open(evaluation_path, 'w') as f:
        json.dump(all_messages, f, indent=4)

if __name__ == "__main__":
    main()














