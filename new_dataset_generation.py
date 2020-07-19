import json
import os
from functools import partial
from typing import List, Callable, Set, Iterable, Union, Tuple

import spacy
from numpy.random import randint, geometric

from io_util import read_file


def masking_by_id(token_list: List[str], id_to_mask: List[int]):
    return [token if idx not in id_to_mask else "[MASK]" for idx, token in enumerate(token_list)]


def deletion_by_id(token_list: List[str], id_to_mask: List[int]):
    return [token for idx, token in enumerate(token_list) if idx not in id_to_mask]

def filter_id_to_mask(tokenized_sent, idx_to_mask, upper_bound):
    def generate_forbidden_indices(sent):
        for idx, token in enumerate(sent):
            if token.pos_ == "ADJ" or token.pos_ == "ADV" or token.dep_ == "aux" or token.dep_ == "det":
                yield idx

    forbidden_indices = set(generate_forbidden_indices(tokenized_sent))
    new_id_to_mask = list()
    for ele in idx_to_mask:
        if ele not in forbidden_indices:
            new_id_to_mask.append(ele)
        else:
            counter = 0
            new_index = randint(0, upper_bound)
            while new_index in forbidden_indices and counter < 20:
                new_index = randint(0, upper_bound)
                counter += 1
            new_id_to_mask.append(new_index)
    return new_id_to_mask

def random_corrupting(tokenized_sent: List[str], process_func: Callable[[List[str], Set[int]], List[str]],
                      probablity=0.2,
                      exists_forbidden_indices=False):
    upper_bound = len(tokenized_sent)
    mask_size = upper_bound * probablity
    mask_size = int(mask_size) + 1 if mask_size > int(mask_size) else int(mask_size)
    idx_to_mask = randint(0, upper_bound, mask_size).tolist()
    new_idx_to_mask = idx_to_mask if not exists_forbidden_indices else filter_id_to_mask(tokenized_sent, idx_to_mask, upper_bound)
    idx_to_mask_set = set(new_idx_to_mask)
    return process_func(tokenized_sent, idx_to_mask_set)


def span_corrupting(tokenized_sent: List[str], max_length: int = 5, prob: float = 0.25):
    max_length = min(len(tokenized_sent) // 2 + 1, max_length)
    span_length = geometric(prob)
    while span_length > max_length or span_length < 2:
        span_length = int(geometric(prob))

    start_point = randint(0, len(tokenized_sent) - span_length + 1)
    return tokenized_sent[: start_point + 1] + tokenized_sent[start_point + span_length:]


def word_reordering(tokenized_sent: List[str]):
    if len(tokenized_sent) < 3:
        print(tokenized_sent)
    reordering_point = randint(1, len(tokenized_sent) - 1)
    return tokenized_sent[reordering_point:] + tokenized_sent[: reordering_point]


def generate_negative_candidates(triplets_list: Iterable[Union[Tuple[str, str, str], List[str]]], tokenizer,
                                 output_path: str):
    random_deletion = partial(random_corrupting, probablity=0.25, process_func=deletion_by_id, exists_forbidden_indices=True)
    random_masking = partial(random_corrupting, probablity=0.25, process_func=masking_by_id, exists_forbidden_indices=True)
    transformation_list = [random_deletion, random_masking, span_corrupting, word_reordering]
    with open(os.path.join(output_path), mode="w+") as out_file:
        for ele in triplets_list:
            hypo, premise, neg_cand = ele
            output_dict = {"hypothesis": hypo,
                           "premise": premise,
                           "negative_candidates": [neg_cand]}

            tokenized_premise = [ele for ele in tokenizer(premise)]
            for corrupt_func in transformation_list:
                result = [ele.text if not isinstance(ele, str) else ele for ele in corrupt_func(tokenized_premise)]
                output_dict["negative_candidates"].append(" ".join(result))
            out_file.write(json.dumps(output_dict) + "\n")


if __name__ == '__main__':
    input_dir = "/home/zxj/Data/relation_based_analogy/input"
    input_path = os.path.join(input_dir, "adjective_compositionality.txt")
    input_iter = read_file(input_path, preprocess=lambda x: x.strip().split("\t"))
    outpath = os.path.join(input_dir, "adjective_analogy.txt")
    nlp = spacy.load("en_core_web_sm")
    generate_negative_candidates(input_iter, nlp, outpath)
