import json
import torch
import timeit
from tqdm import tqdm

from pydantic import BaseModel
from transformers import set_seed, pipeline
import outlines
from outlines.models import Transformers
from outlines.integrations.transformers import JSONPrefixAllowedTokens


DEVICE = torch.device("cuda", 3) if torch.cuda.is_available() else "cpu"
K = 500


def load_model(model_path):
    class Answer(BaseModel):
        answer: bool
    
    t0 = timeit.default_timer()
    pipe = pipeline("question-answering", model=model_path, device=3)
    prefix_allowed_tokens_fn = JSONPrefixAllowedTokens(
        schema=Answer, tokenizer_or_pipe=pipe, whitespace_pattern=r" ?"
    )
    print(f"Loaded model in {timeit.default_timer() - t0:.2f}s")

    return pipe, prefix_allowed_tokens_fn



def inference(model, config, text, question):
    prompt = {
        "question": question,
        "context": text
    }
    response = model(
        prompt,
        return_full_text=False,
        do_sample=False,
        max_new_tokens=10,
        prefix_allowed_tokens_fn=config,
    )
    
    return response['answer']


def process_subject(model, config, file_path, k, target_subject):
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
            text = f"{base_context} {attr_desc} {subject} does not have {attr_title} (False). {subject} has {attr_title} (True)"

            ans = inference(model, config, text, question)

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

    model, config = load_model("deepset/roberta-large-squad2")

    for position in titles:
        path = f"./same_gender/prompts/{position}_prompts.json"

        female_rst = process_subject(model, config, path, K, 'subj1')
        male_rst = process_subject(model, config, path, K, 'subj2')
        rst = female_rst + male_rst

        with open(f"./same_gender/roberta_step1/{position}_results.json", 'w') as fout:
            json.dump(rst, fout)

if __name__ == "__main__":
    main()
