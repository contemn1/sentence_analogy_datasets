import json
import logging
import os
import re
import sys

import spacy

from IOUtil import read_file


def filter_topic(file_path):
    category_pattern = re.compile(r"^: ")
    contents = read_file(file_path, preprocess=lambda x: x.strip())
    for ele in contents:
        if category_pattern.search(ele):
            print(ele)


def construct_dict(root_dir, file_name, output_file_name):
    file_path = os.path.join(root_dir, file_name)
    category_pattern = re.compile(r":[a-zA-Z0-9_\- ]+")

    dicts_per_category = dict()
    result_dict = dict()
    current_key = None
    try:
        with open(file_path, encoding="utf-8") as file:
            for sentence in file:
                result = category_pattern.search(sentence)
                if result:
                    if current_key:
                        dicts_per_category[current_key] = result_dict.copy()
                    current_key = result.group()
                    result_dict.clear()
                else:
                    cap_country_list = [ele.lower() for ele in sentence.strip().split(" ")]
                    if len(cap_country_list) != 4:
                        print(cap_country_list)
                        continue
                    else:
                        cap1, country1, cap2, country2 = cap_country_list
                        if cap1 not in result_dict:
                            result_dict[cap1] = country1
                        if cap2 not in result_dict:
                            result_dict[cap2] = country2

            dicts_per_category[current_key] = result_dict

            result_path = os.path.join(root_dir, output_file_name)
            with open(result_path, "w+") as res_file:
                json.dump(dicts_per_category, res_file)

    except IOError as err:
        logging.error("Failed to open file {0}".format(err))
        sys.exit(1)


def sentences_verb_adj(sentence, parser):
    doc = parser(sentence)
    for token in doc:
        if token.pos_ == "ADJ" and token.dep_ == "acomp" and token.head.dep_ == "ROOT":
            return sentence + "\t" + token.text + "\n"

    return ""


def find_opposite_pairs(dict_path, sent_list, output_path):
    pattern_should_filter = re.compile(r"makes? sure")
    with open(dict_path, mode="r") as json_file:
        opposite_dict = json.load(json_file)[": gram2-opposite"]
        inverse_opposite_dict = {value: key for key, value in opposite_dict.items()}

    with open(output_path, mode="w+") as output_file:
        for ele in sent_list:
            sent, adj = ele
            if pattern_should_filter.search(sent):
                continue

            negation_adj = "not " + adj
            if negation_adj in sent:
                sent = re.sub(negation_adj, adj, sent)

            if adj in opposite_dict:
                opposite = opposite_dict[adj]
                new_sent = re.sub(adj, opposite, sent)
                output_file.write(adj + "\t" + sent + "\t" + new_sent + "\n")

            if adj in inverse_opposite_dict:
                opposite = inverse_opposite_dict[adj]
                new_sent = re.sub(adj, opposite, sent)
                output_file.write(adj + "\t" + new_sent + "\t" + sent + "\n")


if __name__ == '__main__':
    mnli_path = "/home/zxj/Data/multinli_1.0"
    input_path = os.path.join(mnli_path, "multinli_1.0_train_sents.txt")
    sent_list = set(sent for sent_tuple in read_file(input_path, preprocess=lambda x: x.strip().split("\t")) for sent in
                    sent_tuple)

    nlp = spacy.load("en_core_web_sm")
    bad_comparative = {"better", "more", "less", "worse"}
    for sentence in sent_list:
        if " than " not in sentence:
            continue

        doc = nlp(sentence)
        for token in doc:
            if token.text == "than" and token.dep_ == "prep" and token.head.tag_ == "JJR":
                head_node = token.head
                if head_node.text.lower() in bad_comparative:
                    break

                modify_child = [child for child in head_node.children if child.dep_ == "advmod"]
                sent_index = head_node.idx if not modify_child else modify_child[0].idx
                new_sentence = sentence[:sent_index] + head_node.lemma_ + "."
                print(head_node.lemma_ + "\t" + head_node.text + "\t" + new_sentence + "\t" + sentence)
