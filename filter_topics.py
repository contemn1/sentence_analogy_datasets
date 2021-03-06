import io
import json
import logging
import os
import pickle
import re
import string
import sys

import spacy

from io_util import output_list_to_file, read_file


def filter_topic(file_path):
    category_pattern = re.compile(r"^: ")
    contents = read_file(file_path, preprocess=lambda x: x.strip())
    for ele in contents:
        if category_pattern.search(ele):
            print(ele)


def construct_dict(root_dir, file_name, output_file_name, in_lower_case=False):
    file_path = os.path.join(root_dir, file_name)
    category_pattern = re.compile(r":[a-zA-Z0-9_\- ]+")

    dicts_per_category = dict()
    result_dict = dict()
    current_key = None
    try:
        with io.open(file_path, encoding="utf-8") as input_file:
            for sentence in input_file:
                result = category_pattern.search(sentence)
                if result:
                    if current_key:
                        new_current_key = re.sub("^: ", "", current_key)
                        dicts_per_category[new_current_key] = result_dict.copy()
                    current_key = result.group()
                    result_dict.clear()
                else:
                    cap_country_list = [ele for ele in sentence.strip().split(" ")]
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


def create_opposite_dataset(dict_path, sent_list, output_path):
    pattern_should_filter = re.compile(r"makes? sure")
    with open(dict_path, mode="r") as json_file:
        opposite_dict = json.load(json_file)[": gram2-opposite"]
        inverse_opposite_dict = {value: key for key,
                                                value in opposite_dict.items()}

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


def create_comparative_dataset(sent_list, parser):
    bad_comparative = {"better", "more", "less", "worse"}
    for sentence in sent_list:
        if " than " not in sentence:
            continue

        doc = parser(sentence)
        for token in doc:
            if token.text == "than" and token.dep_ == "prep" and token.head.tag_ == "JJR":
                head_node = token.head
                if head_node.text.lower() in bad_comparative:
                    break

                modify_child = [
                    child for child in head_node.children if child.dep_ == "advmod"]
                sent_index = head_node.idx if not modify_child else modify_child[0].idx
                new_sentence = sentence[:sent_index] + head_node.lemma_ + "."
                print(head_node.lemma_ + "\t" + head_node.text +
                      "\t" + new_sentence + "\t" + sentence)


def create_plural_dataset(sent_list, parser):
    for sentence in sent_list:
        doc = parser(sentence)
        for token in doc:
            if token.tag_ == "NNS" and token.dep_ == "dobj" and token.text[-1] == "s":
                num_child = [
                    child for child in token.children if child.dep_ == "nummod"]
                if num_child:
                    sub_children = [
                        child for child in num_child[0].children if child.dep_ == "compound"]
                    text_to_replace = num_child[0].text if not sub_children else sub_children[0].text + " " + num_child[
                        0].text
                    new_sentence = re.sub(token.text, token.lemma_, sentence)
                    new_sentence = re.sub(text_to_replace, "one", new_sentence)
                    print(token.text + "\t" + token.lemma_ +
                          "\t" + new_sentence + "\t" + sentence)


def main():
    mnli_path = "/home/zxj/Data/multinli_1.0"
    input_path = os.path.join(mnli_path, "multinli_1.0_train_sents.txt")
    sent_list = set(sent for sent_tuple in read_file(input_path, preprocess=lambda x: x.strip().split("\t")) for sent in
                    sent_tuple)

    nlp = spacy.load("en_core_web_sm")
    dict_path = os.path.join(mnli_path, "word-pairs-per-category.json")

    plural_verb_dict = json.load(
        open(dict_path, encoding="utf-8"))[": gram9-plural-verbs"]
    for sentence in sent_list:
        doc = nlp(sentence)
        for token in doc:
            if token.dep_ == "nsubj" and token.head.dep_ == "ROOT" and token.tag_ == "NNP":
                root_node = token.head
                root_text = root_node.text
                if root_node.tag_ == "VB" and root_text in plural_verb_dict:
                    child_aux = [
                        child for child in root_node.children if child.dep_ == "aux"]
                    if child_aux:
                        child_negation = [
                            child for child in root_node.children if child.dep_ == "neg"]
                        if not child_negation:
                            words_to_delete = child_aux[0].text
                        elif child_negation[0].text == "not":
                            words_to_delete = child_aux[0].text + \
                                              " " + child_negation[0].text

                        else:
                            words_to_delete = child_aux[0].text + \
                                              child_negation[0].text
                        plural_verb = plural_verb_dict[root_text]
                        new_sentence = re.sub(words_to_delete, "", sentence)
                        new_sentence = re.sub(
                            root_text, plural_verb, new_sentence)
                        new_sentence = re.sub("\s+", " ", new_sentence)
                        print(root_text + "\t" + plural_verb +
                              "\t" + sentence + "\t" + new_sentence)
                        break


def get_sentences_with_certain_words(file_path, dict_path, category_name, output_path, capitalize=False):
    sentence_iterator = read_file(
        file_path, preprocess=lambda x: x.strip().split("\t"))
    sentence_set = set([sent for arr in sentence_iterator for sent in arr])

    with open(dict_path, "r") as category_dict:
        category_list = json.load(category_dict)[category_name]
    key_value_list = []
    for key, value in category_list.items():
        if capitalize:
            key = string.capwords(key)
            value = string.capwords(value)

        key_value_list.append(" {0} ".format(key))
        if value.lower() != "real":
            key_value_list.append(" {0} ".format(value))

    pattrn_str = "|".join(key_value_list)
    pattern = re.compile(pattrn_str)
    new_senence_set = [sent for sent in sentence_set if pattern.search(sent)]
    output_list_to_file(output_path, new_senence_set)


def load_gensen_vocab():
    with open("/media/zxj/sent_embedding_data/gensen_models/nli_large_bothskip_vocab.pkl") as vocab_file:
        model_vocab = pickle.load(vocab_file)
        word2id = [word.decode("utf-8") for word in model_vocab['word2id']]
        word2id = set(word2id)
    return word2id


def calcualte_overlap(dict_dir, dict_name, vocabulary):
    dict_path = os.path.join(dict_dir, dict_name)
    category_vocab = list(
        read_file(dict_path, preprocess=lambda x: x.strip().split("\t")))
    new_vocab = []
    oov_pairs = []
    for first, second in category_vocab:
        '''
        if dict_name != "family_words.txt":
            first = string.capwords(first)
        if dict_name not in {"family_words.txt", "currency_words.txt"}:
            second = string.capwords(second)
        '''
        if first in vocabulary and second in vocabulary:
            new_vocab.append(first + "\t" + second)
        else:
            oov_pairs.append(first + "\t" + second)

    overlap_rate = float(len(new_vocab)) / float(len(category_vocab))
    return new_vocab, oov_pairs, overlap_rate


def calculate_overlap_per_list(dict_dir, dict_name_list, vocabulary):
    for name in dict_name_list:
        new_vocab, oov_vocab, over_lap_rate = calcualte_overlap(
            dict_dir, name, vocabulary)
        name_without_suffix = re.sub(".txt$", "", name)
        if over_lap_rate - 1.0 < 0:
            print("oov word  pairs in {0} dataset are:".format(
                name_without_suffix))
            for ele in oov_vocab:
                print(ele)
        else:
            print("no oov word pairs in {0} dataset".format(
                name_without_suffix))


def generate_certain_category_dict(file_path, output_dir):
    with io.open(file_path) as input_file:
        json_dict = json.load(input_file)

    for key, value in json_dict.items():
        if not re.match(r"gram(?!6)\d-", key):
            key = re.sub("gram6-", "", key)
            output_path = os.path.join(output_dir, "{0}_words.txt".format(key))
            with io.open(output_path, mode="w+") as output_file:
                for key2, value2 in value.items():
                    output_file.write(key2 + "\t" + value2 + "\n")


def main():
    root_dir = "/home/zxj/Data/multinli_1.0"
    file_name = "multinli_1.0_train_sents.txt"
    file_path = os.path.join(root_dir, file_name)
    dict_path = os.path.join(root_dir, "word-pairs-per-category.json")
    get_sentences_with_certain_words(file_path, dict_path, "currency", os.path.join(root_dir, "sents_with_currency"))


if __name__ == '__main__':
    root_dir = "/home/zxj/Data/multinli_1.0"
    file_name = "word-pairs-per-category.json"
    file_path = os.path.join(root_dir, file_name)
    output_dir = "/home/zxj/Data/multinli_1.0/dict"
    generate_certain_category_dict(file_path, output_dir)
