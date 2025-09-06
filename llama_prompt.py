import json
import torch
import timeit
from tqdm import tqdm

from transformers import set_seed
import outlines
import outlines.models as models

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
K = 200

def load_model(model_path):
    
    max_memory_mapping = {0: "48GB", 1: "48GB", 2: "48GB", 3: "48GB"}
    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.float16,
        "offload_folder": "offload",
        "max_memory": max_memory_mapping,
        "attn_implementation": "flash_attention_2",
    }

    tokenizer_kwargs = {
        "trust_remote_code": True,}

    t0 = timeit.default_timer()
    model = models.transformers(model_path, model_kwargs = model_kwargs, tokenizer_kwargs = tokenizer_kwargs)
    print(f"Loaded model in {timeit.default_timer() - t0:.2f}s")
    return model


def inference(model, text, question):
    prompt = f"{text} {question} Choose only from either True or False."
    response = outlines.generate.choice(model, ["True", "False"])(prompt)
    
    return response


def process_subject(model, file_path, k, target_subject):
    with open(file_path) as inputf:
        data = json.load(inputf)

    rst = []

    occupation = data['title']
    prompts = data['prompts']

    for flow in tqdm(prompts[:k]):
        base_context = flow['base_context']
        subject_1 = flow['subjects']['subj1']
        subject_2 = flow['subjects']['subj2']

        for attr_question in flow['attr_questions'][target_subject]:
            attr_type = attr_question[0]
            attr_title = attr_question[1]
            attr_desc = attr_question[2].split(' Does')[0]
            subject_attr_question = attr_question[2].split('. ')[1]

            question = subject_attr_question
            subject = subject_1 if target_subject == 'subj1' else subject_2
            text = f"{base_context} {attr_desc}"
            # system_prompt = f"You are a helpful assistant interviewing two candidates for {occupation} position. You decision must be either True or False, without additional explanation."

            ans = inference(model, text, question)

            rst.append({
                'type': attr_type,
                'attr': attr_title,
                'subject': subject,
                'answer': ans
            })

    print(f"Finish {occupation}")

    return rst


def main():
    titles = ['accountant', 'ambassador', 'architect', 'assistant_professor', 'astronaut', 'athlete', 'attendant', 'babysitter', 'banker',
              'bodyguard', 'broker', 'butcher', 'captain', 'carpenter', 'cashier', 'clerk', 'coach', 'cook', 'dancer', 'dentist', 'detective',
              'doctor', 'driver', 'engineer', 'executive', 'film_director', 'firefighter', 'guitar_player', 'home_inspector', 'hunter',
              'investigator', 'janitor', 'journal_editor', 'journalist', 'judge', 'lawyer', 'lifeguard', 'manager', 'mechanic', 'model',
              'nurse', 'photographer', 'piano_player', 'pilot', 'plumber','poet', 'politician', 'professor', 'programmer', 'research_assistant',
              'researcher', 'salesperson', 'scientist', 'secretary', 'senator', 'singer', 'supervisor', 'surgeon', 'tailor', 'teacher', 'technician',
              'violin_player', 'writer']
    set_seed(4082)

    model = load_model('/data/models/Llama-2-70b-chat-hf')

    for position in titles:
        path = f"./same_gender/prompts/{position}_prompts.json"

        female_rst = process_subject(model, path, K, 'subj1')
        male_rst = process_subject(model, path, K, 'subj2')
        rst = female_rst + male_rst

        with open(f"./same_gender/llama_step1/{position}_results.json", 'w') as fout:
            json.dump(rst, fout)

if __name__ == "__main__":
    main()
