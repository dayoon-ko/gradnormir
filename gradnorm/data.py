import math
import json 
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
    def __init__(self, 
                 args: DataArguments, 
                 batch_size: int, 
                 seed: int, 
                 process_index: int=0, 
                 num_processes: int=1,
                 logger=None,
                 ):
        train_datasets = []
        each_data_inxs = []
        batch_size_inxs = []
        pqloss_flag = []
        cur_all_num = 0
        self.logger = logger
        
        SMALL_THRESHOLD = args.small_threshold
        DROP_THRESHOLD = args.drop_threshold
        
        context_feat_meta = datasets.Features({
            '_id': datasets.Value('string'),
            'recall': datasets.Value('float'),
            'n-query': datasets.Value('int64'),
            'query': datasets.Value('string'),
            'pos': datasets.Sequence(datasets.Value('string')),
            'neg': datasets.Sequence(datasets.Value('string'))
        })
        assert isinstance(args.train_data, list) and len(args.train_data) >= 1
        
        #if dist.get_rank() == 0:
        #    self.print_batch_size(batch_size=batch_size, train_group_size=args.train_group_size)
        
        for file_path in args.train_data:
            
            small_datasets = []
            small_batch_size = math.inf
            
            # Add `parallel_` in `data_dir` to indicate that this dataset is parallel corpus
            flag = 'parallel_' in file_path
            if not (file_path.endswith('.json') or file_path.endswith('.jsonl')):
                continue
            if dist.get_rank() == 0:
                self.logger.info(f'loading data from {file_path} ...')
            temp_dataset = datasets.load_dataset('json', data_files=file_path, split='train', cache_dir=args.cache_path, features=context_feat_meta)
            selected_indices = self.get_remaining_indices(args.logging_pth, file_path)
            temp_dataset = temp_dataset.select(selected_indices)
            #temp_dataset = temp_dataset.sort('_id')
            if len(temp_dataset) == 0:
                continue
            elif len(temp_dataset) < SMALL_THRESHOLD:
                small_datasets.append(temp_dataset)
                small_batch_size = min(small_batch_size, self.get_file_batch_size(file_path, batch_size, train_group_size=args.train_group_size))
            else:
                if args.max_example_num_per_dataset is not None and len(temp_dataset) > args.max_example_num_per_dataset:
                    temp_dataset = temp_dataset.select(
                        random.sample(list(range(len(temp_dataset))), args.max_example_num_per_dataset))
                train_datasets.append(temp_dataset)
                each_data_inxs.append(np.arange(len(temp_dataset)) + cur_all_num)
                cur_all_num += len(temp_dataset)
                batch_size_inxs.append(self.get_file_batch_size(file_path, batch_size, train_group_size=args.train_group_size))
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
        self.step = 0
        self.refresh_epoch()
    
    def get_remaining_indices(self, logging_pth: str, file_path: str):        
        # Read already done results
        lines = open(logging_pth).readlines()
        done_ids = set()
        for l in lines: #[:-32]:
            try:
                sid = l.index("doc_id: ") + len("doc_id: ") 
                eid = sid
                done_ids.add(l[sid:].strip())
            except Exception as e:
                continue
        self.logger.info(f"Already done: {len(done_ids)} datapoints")
        
        # Read dataset and filter already done
        js = json.load(open(file_path))
        indices = [i for i, item in enumerate(js) if item["_id"] not in done_ids]
        self.logger.info(f"Left: {len(indices)} from {len(js)}")
        
        return indices

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
        # Dynamically adjust batch size
        batch_datas = []
        for dataset_inx in self.datasets_inxs:
            cur_batch_size = self.batch_size_inxs[dataset_inx]*self.num_processes
            flag = self.pqloss_flag[dataset_inx]
            for start_index in range(0, len(self.each_data_inxs[dataset_inx]), cur_batch_size):
                # judge the last batch's length
                if len(self.each_data_inxs[dataset_inx]) - start_index < 2 * self.num_processes:
                    break
                batch_datas.append((self.each_data_inxs[dataset_inx][start_index:start_index+cur_batch_size], flag))
        self.batch_datas = batch_datas
        self.step = 0

    def __getitem__(self, idx):  
        query = self.dataset[idx]['query']
        passages = self.dataset[idx]['pos'] + self.dataset[idx]['neg']
        meta = self.dataset[idx]
        self.logger.info("doc_id: %s", meta["_id"])
        self.logger.info("recall: %s", meta["recall"])
        del meta["query"], meta["pos"], meta["neg"]
        return query, passages, None, None

    def create_batch_data(self, batch_raw_data):
        queries, passages = [], []
        teacher_scores = []
        for i in range(len(batch_raw_data['query'])):            
            queries.append(batch_raw_data['query'][i])
            passages.append(batch_raw_data['pos'][i][0])
            passages.extend(batch_raw_data["neg"][i])

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

    def __call__(self, features):
        query = [f[0] for f in features]
        passage = [f[1] for f in features]
        
        teacher_scores = None
        if len(features[0]) > 2:
            teacher_scores = [f[2] for f in features]
            if teacher_scores[0] is None:
                teacher_scores = None
            else:
                teacher_scores = torch.FloatTensor(teacher_scores)
        
        flag = None
        if len(features[0]) == 4:
            flag = [f[3] for f in features][0]
            
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer(
            query,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer(
            passage,
            # padding='max_length',     # used for adjusting the batch size in `get_file_batch_size()`
            padding=True,
            truncation=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )
        if teacher_scores is not None:
            teacher_scores = teacher_scores.reshape((len(q_collated['input_ids']), -1))
        return {"query": q_collated, "passage": d_collated, "teacher_scores": teacher_scores, "bi_directions": flag}



