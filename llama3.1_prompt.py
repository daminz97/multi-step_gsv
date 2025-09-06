import json
import asyncio

from tqdm import tqdm
from tqdm.asyncio import tqdm as async_tqdm

from openai import AsyncOpenAI, OpenAI
from pydantic import BaseModel
from typing import Literal

K = 50
# BATCH_SIZE = 50


class Answer(BaseModel):
    ans: Literal['True', 'False']
    

def call_api_async(text, question):    
    prompt = f"{text} {question} Choose only from either True or False."
    client = OpenAI(
        api_key="****",
        base_url="http://localhost:8080/v1",
        timeout=60,
        max_retries=3
    )
    
    completion = client.beta.chat.completions.parse(
        model="/data/models/Meta-Llama-3.1-70B-Instruct",
        messages=[
            {"role": "system", "content": f"You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.7,
        response_format=Answer,
    )
    
    return completion.choices[0].message.parsed.ans



def process_attr_question(flow, target_subject, attr_question):
    base_context = flow['base_context']
    subject_1 = flow['subjects']['female']
    subject_2 = flow['subjects']['male']
    
    attr_type = attr_question[0]
    attr_title = attr_question[1]
    attr_desc = attr_question[2].split(' Does')[0]
    subject_attr_question = attr_question[2].split('. ')[1]

    question = subject_attr_question
    subject = subject_1 if target_subject == 'female' else subject_2
    text = f"{base_context} {attr_desc}"
    
    response = call_api_async(text, question)
        
    return {
        'type': attr_type,
        'attr': attr_title,
        'subject': subject,
        'answer': response
    }

def batch_annotate(prompts, target_subject):
    tasks = []
    
    for flow in tqdm(prompts[:K]):
        for attr_question in flow['attr_questions'][target_subject]:
            tasks.append(process_attr_question(flow, target_subject, attr_question))
    
    # results = []
    # for i in tqdm(range(0, len(tasks), BATCH_SIZE)):
    #     batch = tasks[i:i+BATCH_SIZE]
    #     batch_results = asyncio.gather(*batch)
    #     results.extend(batch_results)
    
    return tasks


def batch_inference():
    titles = ['journal_editor', 'journalist', 'judge', 'lawyer', 'lifeguard', 'manager', 'mechanic', 'model',
              'nurse', 'photographer', 'piano_player', 'pilot', 'plumber','poet', 'politician', 'professor', 'programmer', 'research_assistant',
              'researcher', 'salesperson', 'scientist', 'secretary', 'senator', 'singer', 'supervisor', 'surgeon', 'tailor', 'teacher', 'technician',
              'violin_player', 'writer']
    
    
    for position in titles:
        # path = f"./same_gender/prompts/{position}_prompts.json"
        path = f"./diff_gender/prompts/{position}_prompts.json"
        
        with open(path) as inputf:
            data = json.load(inputf)
            
        occupation = data['title']
        prompts = data['prompts']
        
        female_rst = batch_annotate(prompts, 'female')
        male_rst = batch_annotate(prompts, 'male')
        rst = female_rst + male_rst
        
        print(f"Finish {occupation}")
        
        # with open(f"./same_gender/llama31_step1/{position}_results.json", 'w') as fout:
        #     json.dump(rst, fout)
        
        with open(f"./diff_gender/llama31_step1/{position}_results.json", 'w') as fout:
            json.dump(rst, fout)



if __name__ == "__main__":
    batch_inference()