import math
import os.path
import random
from dataclasses import dataclass
import torch
import numpy as np
import datasets
from pprint import pprint
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
import torch.distributed as dist

from arguments import DataArguments


class SameDatasetTrainDataset(Dataset):
    """Dataset to yield a batch of data at one time. All samples in the same batch comes from the same task.
    """
    def __init__(self, args: DataArguments, batch_size: int, seed: int, process_index: int=0, num_processes: int=1):
        train_datasets = []
        each_data_inxs = []
        batch_size_inxs = []
        pqloss_flag = []
        cur_all_num = 0
        
        SMALL_THRESHOLD = args.small_threshold
        DROP_THRESHOLD = args.drop_threshold
        
        context_feat_meta = datasets.Features({
            '_id': datasets.Value('string'),
            'recall': datasets.Value('string'),
            'precision': datasets.Value('string'),
            'f1': datasets.Value('string'),
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string'))
        })
        context_feat_kd = datasets.Features({
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string')),
            'pos_scores': datasets.Sequence(datasets.Value('float')),
            'neg_scores': datasets.Sequence(datasets.Value('float')),
        })
        assert isinstance(args.train_data, list) and len(args.train_data) >= 1
        
        if dist.get_rank() == 0:
            self.print_batch_size(batch_size=batch_size, train_group_size=args.train_group_size)
        
        for data_dir in args.train_data:
            if not os.path.isdir(data_dir):
                raise FileNotFoundError(f"{data_dir} is a file, not a directionary")
            
            small_datasets = []
            small_batch_size = math.inf
            
            # Add `parallel_` in `data_dir` to indicate that this dataset is parallel corpus
            flag = 'parallel_' in data_dir
            for file in os.listdir(data_dir):
                if not (file.endswith('.json') or file.endswith('.jsonl')):
                    continue
                
                file_path = os.path.join(data_dir, file)
                if dist.get_rank() == 0:
                    print(f'loading data from {file_path} ...')
                try:
                    temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat_meta)
                except:
                    temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat_kd)
                    if not args.knowledge_distillation:
                        temp_dataset = temp_dataset.remove_columns(['pos_scores', 'neg_scores'])
                
                if len(temp_dataset) == 0:
                    continue
                elif len(temp_dataset) < SMALL_THRESHOLD:
                    small_datasets.append(temp_dataset)
                    small_batch_size = min(small_batch_size, self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                else:
                    if args.max_example_num_per_dataset is not None and len(temp_dataset) > args.max_example_num_per_dataset:
                        temp_dataset = temp_dataset.select(
                            random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                    train_datasets.append(temp_dataset)
                    each_data_inxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                    cur_all_num += len(temp_dataset)
                    batch_size_inxs.append(self.get_file_batch_size(file, batch_size, train_group_size=args.train_group_size))
                    pqloss_flag.append(flag)
            
            if len(small_datasets) > 0:
                small_dataset = datasets.concatenate_datasets(small_datasets)
                if len(small_dataset) >= DROP_THRESHOLD:
                    train_datasets.append(small_dataset)
                    each_data_inxs.append(np.arange(len(small_dataset)) + cur_all_num)
                    cur_all_num += len(small_dataset)
                    batch_size_inxs.append(small_batch_size)
                    pqloss_flag.append(flag)
        
        self.dataset = datasets.concatenate_datasets(train_datasets)
        self.each_data_inxs = each_data_inxs
        self.datasets_inxs = np.arange(len(each_data_inxs))
        self.batch_size_inxs = batch_size_inxs
        self.pqloss_flag = pqloss_flag
        
        self.process_index = process_index
        self.num_processes = num_processes
        self.args = args
        self.shuffle_ratio = args.shuffle_ratio
        
        self.deterministic_generator = np.random.default_rng(seed)
        self.step = 0
        self.refresh_epoch()
    
    def print_batch_size(self, batch_size: int, train_group_size: int):
        length_list = ['0-500', '500-1000', '1000-2000', '2000-3000', '3000-4000', '4000-5000', '5000-6000', '6000-7000', '7000-inf']
        batch_size_dict = {
            k: self.get_file_batch_size(f"len-{k}.jsonl", batch_size, train_group_size) for k in length_list
        }
        batch_size_list = [
            f'{length}: {batch_size_dict[length]}' for length in length_list
        ]
        print("=========================")
        print("Batch Size Dict:")
        pprint(batch_size_list)
        print("=========================")
    
    @staticmethod
    def get_file_batch_size(file: str, batch_size: int, train_group_size: int):
        if train_group_size == 8:
            # 80GB
            if 'len-0-500.jsonl' in file:
                return 48
            elif 'len-500-1000.jsonl' in file:
                return 32
            elif 'len-1000-2000.jsonl' in file:
                return 20
            elif 'len-2000-3000.jsonl' in file:
                return 18
            elif 'len-3000-4000.jsonl' in file:
                return 14
            elif 'len-4000-5000.jsonl' in file:
                return 14
            elif 'len-5000-6000.jsonl' in file:
                return 12
            elif 'len-6000-7000.jsonl' in file:
                return 10
            elif 'len-7000-inf.jsonl' in file:
                return 8
            else:
                return batch_size
        elif train_group_size == 1:
            # 80GB
            if 'len-0-500.jsonl' in file:
                return 700
            elif 'len-500-1000.jsonl' in file:
                return 570
            elif 'len-1000-2000.jsonl' in file:
                return 388
            elif 'len-2000-3000.jsonl' in file:
                return 288
            elif 'len-3000-4000.jsonl' in file:
                return 224
            elif 'len-4000-5000.jsonl' in file:
                return 180
            elif 'len-5000-6000.jsonl' in file:
                return 157
            elif 'len-6000-7000.jsonl' in file:
                return 128
            elif 'len-7000-inf.jsonl' in file:
                return 104
            else:
                return batch_size
        else:
            return batch_size
    
    def refresh_epoch(self):
        print(f'---------------------------*Rank {self.process_index}: refresh data---------------------------')
        self.deterministic_generator.shuffle(self.datasets_inxs)
        # Dynamically adjust batch size
        batch_datas = []
        for dataset_inx in self.datasets_inxs:
            self.deterministic_generator.shuffle(self.each_data_inxs[dataset_inx])
            cur_batch_size = self.batch_size_inxs[dataset_inx]*self.num_processes
            flag = self.pqloss_flag[dataset_inx]
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                # judge the last batch's length
                if len(self.each_data_inxs[dataset_inx]) - start_index < 2 * self.num_processes:
                    break
                batch_datas.append((self.each_data_inxs[dataset_inx][start_index:start_index+cur_batch_size], flag))
        self.deterministic_generator.shuffle(batch_datas)
        self.batch_datas = batch_datas
        self.step = 0

    def __getitem__(self, _):  
        batch_indices, pqloss_flag = self.batch_datas[self.step]
        cur_batch_size = int(len(batch_indices) / self.num_processes)
        batch_indices = batch_indices[self.process_index * cur_batch_size: (self.process_index + 1) * cur_batch_size]
        batch_data = self.dataset[batch_indices]
        self.step += 1
        queries, passages, teacher_scores = self.create_batch_data(batch_raw_data=batch_data)
        # print('rank, step, flag, query, passage:', dist.get_rank(), self.step, pqloss_flag, queries, passages)
        return queries, passages, teacher_scores, pqloss_flag

    def shuffle_text(self, text):
        if self.shuffle_ratio > 0 and len(text) > 100 and random.random() < self.shuffle_ratio:
            split_text = []
            chunk_size = len(text)//3 + 1
            for i in range(0, len(text), chunk_size):
                split_text.append(text[i:i+chunk_size])
            random.shuffle(split_text)
            return " ".join(split_text)
        else:
            return text

    def create_batch_data(self, batch_raw_data):
        queries, passages = [], []
        teacher_scores = []
        for i in range(len(batch_raw_data['query'])):            
            queries.append(batch_raw_data['query'][i])
            
            pos_inx = random.choice(list(range(len(batch_raw_data['pos'][i]))))
            passages.append(self.shuffle_text(batch_raw_data['pos'][i][pos_inx]))
            if 'pos_scores' in batch_raw_data and batch_raw_data['pos_scores'][i] is not None:
                teacher_scores.append(batch_raw_data['pos_scores'][i][pos_inx])
            
            neg_inx_set = list(range(len(batch_raw_data['neg'][i])))
            if len(batch_raw_data['neg'][i]) < self.args.train_group_size - 1:
                num = math.ceil((self.args.train_group_size - 1) / len(batch_raw_data['neg'][i]))
                neg_inxs = random.sample(neg_inx_set * num, self.args.train_group_size - 1)
            else:
                neg_inxs = random.sample(neg_inx_set, self.args.train_group_size - 1)            
            
            if 'neg_scores' in batch_raw_data and batch_raw_data['neg_scores'][i] is not None:
                neg_scores = [(x, batch_raw_data['neg_scores'][i][x]) for x in neg_inxs]
                neg_scores = sorted(neg_scores, key=lambda x:x[1], reverse=True)
                neg_inxs = [x[0] for x in neg_scores]
                teacher_scores.extend([x[1] for x in neg_scores])
                
            negs = [batch_raw_data['neg'][i][x] for x in neg_inxs]
            passages.extend(negs)
            
            if len(teacher_scores) > 0 and len(passages) > 0:
                assert len(teacher_scores) == len(passages)

        if self.args.query_instruction_for_retrieval is not None:
            queries = [self.args.query_instruction_for_retrieval+q for q in queries]
        if self.args.passage_instruction_for_retrieval is not None:
            passages = [self.args.passage_instruction_for_retrieval+p for p in passages]
        
        if len(teacher_scores) == 0:
            teacher_scores = None
        return queries, passages, teacher_scores
    
    def __len__(self):
        return len(self.batch_datas) * self.num_processes


@dataclass
class EmbedCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 512
    passage_max_len: int = 512
    ratio_min: float = 0.5
    ratio_max: float = 0.8

    def __call__(self, features):
        query = [f[0] for f in features]
        if isinstance(query[0], list):
            query = sum(query, [])

        # Tokenize
        q_tokens_all = []
        p_tokens_all = []
        for q in query:
            org_tokens = self.tokenizer(
                q,
                padding=True,
                truncation=True,
                max_length=self.query_max_len,
                return_tensors="pt",
            )
            org_tokens = org_tokens["input_ids"][0]
            # Random crop for query, positive and negative samples
            q_tokens = randomcrop(org_tokens, self.ratio_min, self.ratio_max)
            p_tokens = randomcrop(org_tokens, self.ratio_min, self.ratio_max)
            q_tokens = add_bos_eos(q_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
            p_tokens = add_bos_eos(p_tokens, self.tokenizer.bos_token_id, self.tokenizer.eos_token_id)
            q_tokens_all.append(q_tokens)
            p_tokens_all.append(p_tokens)
        n_tokens_all = [p_tokens_all[:i] + p_tokens_all[i+1:] for i in range(len(query))]
        n_tokens_all = sum(n_tokens_all, [])
        
        # Build mask
        max_length = max(max([len(x) for x in q_tokens_all]), 
                         max([len(x) for x in p_tokens_all]))
        q_tokens_all, q_mask = build_mask(q_tokens_all, max_length)
        p_tokens_all, p_mask = build_mask(p_tokens_all, max_length)
        n_tokens_all, n_mask = build_mask(n_tokens_all, max_length)
        
        #print("Q", q_tokens_all.shape, q_mask.shape)
        #print("P", p_tokens_all.shape, p_mask.shape)
        #print("N", n_tokens_all.shape, n_mask.shape)
        
        # Concat positive and negative samples
        batch_size, max_length = p_tokens_all.shape
        p_tokens_all = p_tokens_all.reshape(batch_size, -1, max_length)
        n_tokens_all = n_tokens_all.reshape(batch_size, -1, max_length)
        p_mask = p_mask.reshape(batch_size, -1, max_length)
        n_mask = n_mask.reshape(batch_size, -1, max_length)
        k_tokens_all = torch.cat([p_tokens_all, n_tokens_all], dim=1)
        k_mask = torch.cat([p_mask, n_mask], dim=1)
        k_tokens_all = k_tokens_all.reshape(-1, max_length)
        k_mask = k_mask.reshape(-1, max_length)

        # Make final dictionary output 
        q_collated = {"input_ids": q_tokens_all, "attention_mask": q_mask}
        k_collated = {"input_ids": k_tokens_all, "attention_mask": k_mask}
        
        return {"query": q_collated, "passage": k_collated, "teacher_scores": None, "bi_directions": None}


def randomcrop(x, ratio_min, ratio_max):
    ratio = random.uniform(ratio_min, ratio_max)
    length = int(len(x) * ratio)
    start = random.randint(0, len(x) - length)
    end = start + length
    crop = x[start:end].clone()
    return crop


def build_mask(tensors, max_length):
    shapes = [x.shape for x in tensors]
    returnmasks = []
    ids = []
    for k, x in enumerate(tensors):
        returnmasks.append(torch.tensor([1] * len(x) + [0] * (max_length - len(x))))
        ids.append(torch.cat((x, torch.tensor([0] * (max_length - len(x))))))
    ids = torch.stack(ids, dim=0).long()
    returnmasks = torch.stack(returnmasks, dim=0).bool()
    return ids, returnmasks


def add_bos_eos(x, bos_token_id, eos_token_id):
    if not isinstance(x, torch.Tensor):
        x = torch.Tensor(x)
    if bos_token_id is None and eos_token_id is not None:
        x = torch.cat([x.clone().detach(), torch.tensor([eos_token_id])])
    elif bos_token_id is not None and eos_token_id is None:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach()])
    elif bos_token_id is None and eos_token_id is None:
        pass
    else:
        x = torch.cat([torch.tensor([bos_token_id]), x.clone().detach(), torch.tensor([eos_token_id])])
    return x