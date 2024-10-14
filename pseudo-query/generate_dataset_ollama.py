'''
conda activate sds; python generate_dataset_ollama.py  -w haring -m llama3.1 -i 0 -p 60000 -ds trec-covid-v2
'''
import os 
import re
import fire
import json
import time
import torch
import argparse
from tqdm import tqdm
from pprint import pprint
from litellm import completion, batch_completion
from transformers import pipeline
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import Accelerator, init_empty_weights, load_checkpoint_and_dispatch
import random 
random.seed(0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-ds", default="trec-covid-v2")
    parser.add_argument("--batch_size", "-bs", type=int, default=1)
    parser.add_argument("--num_iter", default=1)
    parser.add_argument("--temperature", default=0.5)
    parser.add_argument("--max_new_tokens", default=300)
    parser.add_argument('--worker', '-w', type=str, help='Nodename to run the server on', required=True) 
    parser.add_argument('--port', '-p', type=int, help='Port to run the server on', required=True)
    parser.add_argument('--model', '-m', type=str, choices=['llama3', 'llama3:70b', 'llama3.1', 'llama3.1:70b'], help='Model to use', default='llama3.1') 
    parser.add_argument('--chunk_index', '-i', type=int, required=True)
    parser.add_argument("--chunk_size", "-s", type=int, default=10000000)
    return parser.parse_args()


def api_call(args, content):
    res = completion(
        model=f"ollama/{args.model}", 
        messages=[{"content": content, "role": "user"}], 
        api_base=f"http://{args.worker}.snu.vision:{args.port}",
        max_tokens=args.max_new_tokens,
        temperature=args.temperature
    )
    #time.sleep(0.1)
    res = res.choices[0].message.content
    
    return res


def get_qa(args, prompt):

    count = 0
    while count < 100000:
        count += 1
        res = api_call(args, prompt)
        extracted_list = re.findall(r"\{(.*?)\}", res)
        if len(extracted_list) != args.num_iter * 2:
            print(res)
            continue
        qa = [{"question": extracted_list[i],
            "answer": extracted_list[i+1]}
            for i in range(0, args.num_iter*2, 2)]
        return qa if len(qa) > 1 else qa[0]


class AugDataset(Dataset):
    def __init__(self, args):
        start_idx = args.chunk_size * args.chunk_index 
        end_idx = args.chunk_size * (args.chunk_index + 1)
        dataset = [json.loads(i) for i in open(args.input_fn).readlines()][start_idx:end_idx]
        dataset = self._filter_dataset(dataset, args.save_fn)
        self.dataset = dataset 
        self.args = args
    
    def __getitem__(self, idx):
        return self.dataset[idx]

    def __len__(self):
        return len(self.dataset)
    
    def _filter_dataset(self, dataset, save_fn):
        if not os.path.exists(save_fn):
            return dataset 
        dataset = {i["_id"]: i for i in dataset}
        with open(save_fn) as f:
            done = [json.loads(i)["_id"] for i in f.readlines()]
        todo = set(dataset.keys()) - set(done)
        dataset = [dataset[i] for i in todo]
        return dataset
        
            
    def get_prompt(self, input):
        # check whether enough info
        
        content = f"Generate one Q&A pair based on a given context, where the context is understood but NOT DIRECTLY VISIBLE to the person answering the question. " +\
                "The question should cover the main focus of the full context.\n" +\
                "Assume the person answering the question has common sense and is aware of the details and key points in the sentence(s), but the sentence(s) itself is not quoted or referenced directly.\n\n" + \
                f"Sentence(s) : {input['text'][:1000]}\n\n" +\
                "Use the following instructions for generating a Q&A pairs:\n" +\
                "1) Provide one {question}{answer}.\n"+\
                "2) DON’T use phrases such as ‘according to the sentence(s)’ in your question.\n" +\
                "3) DON’T use phrases in the context verbatim.\n" +\
                "4) An answer should be an entity or entities.\n" +\
                "5) Ensure the question can be answered without referring back to the document, assuming domain knowledge.\n" +\
                "6) Ensure the question includes enough context to be understood on its own.\n" +\
                "7) The question should be general enough to be answerable by someone familiar with the topic, not requiring specific details from the context.\n"+\
                "8) If there is not enough information to generate a question, state 'Not enough information to generate a question.'\n\n" +\
                "Be sure to follow the following format and provide a question and answer pair within curly brackets.\n" +\
                "The format is as follows: "
        for _ in range(args.num_iter):
            content += "{Question}{Answer}\n"
        '''
        content = f"Generate one Q&A pair based on a given context, where the context is understood but NOT DIRECTLY VISIBLE to the person answering the question. " +\
                "The question should cover the main focus of the full context.\n" +\
                "Assume the person answering the question has common sense and is aware of the details and key points in the sentence(s), but the sentence(s) itself is not quoted or referenced directly.\n\n" + \
                f"Sentence(s) : {input['text']}\n\n" +\
                "Provide one {question}{answer}.\n"
        '''
        return content
    
    def collate_fn(self, batch):
        prompts = [self.get_prompt(i) for i in batch]
        if len(prompts) > 1:
            return prompts, batch
        else:
            return prompts[0], batch[0]


def main(args):
    
    # Set input and save file name
    args.input_fn = f"datasets/{args.dataset}/corpus_selected.jsonl"
    args.save_fn = f"datasets/{args.dataset}/queries_generated.jsonl" 
    if "70" in args.model:
        args.save_fn = f"datasets/{args.dataset}/queries_generated_70b.jsonl" 
    
    # Load dataset
    dataset = AugDataset(args)
    dataloader = DataLoader(dataset, 
                            batch_size=args.batch_size, 
                            collate_fn=dataset.collate_fn
                            )
    # Run inference
    for batch in tqdm(dataloader):
        prompt, item = batch
        #print(prompt)
        output = get_qa(args, prompt)
        item = {
            "_id": item["_id"],
            "text": output["question"],
            "meta": output
        }
        with open(args.save_fn, "a") as f:
            f.write(json.dumps(item) + "\n")


if __name__ == "__main__":
    args = get_args()
    main(args)
    