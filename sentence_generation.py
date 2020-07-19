import json
import os
import string
import spacy
from io_util import read_file, output_list_to_file


def read_template_dict(template_name, category_name):
    root_dir = "/home/zxj/Data/multinli_1.0"
    template_path = os.path.join(root_dir, template_name)
    dict_path = os.path.join(root_dir, "word-pairs-per-category.json")
    pair_dict = json.load(open(dict_path))[category_name]
    templates = read_file(template_path, preprocess=lambda x: x.strip())
    return templates, pair_dict


def generate_capital_countries(templates, country_dict):
    result_list = []
    for tem in templates:
        for country, city in country_dict.items():
            output_list = [country, city, tem.format(country), tem.format(city)]
            result_list.append("\t".join(output_list))

    return result_list


def generate_currency(templates, currency_dict):
    for idx, temp in enumerate(templates):
        currency_template, country_template = temp.split("\t")  # type: str
        for country, currency in currency_dict.items():
            output_list = [country, currency, country_template.format(country),
                           currency_template.format(currency).capitalize()]
            yield "\t".join(output_list)


def generate_city_in_state(templates, city_dict):
    for idx, temp in enumerate(templates):
        for city, state in city_dict.items():
            if idx >= 4:
                city_template, state_template = temp.split("\t")
            else:
                city_template, state_template = temp, temp

            output_list = [city, state, city_template.format(city), state_template.format(state)]
            yield "\t".join(output_list)


def generate_family(templates, family_dict):
    special_key_list = {"boy", "king", "groom", "prince", "man", "policeman"}
    templates = [temp.split("\t") for temp in templates]
    output_list = []
    for idx, temp in enumerate(templates):
        for key, value in family_dict.items():
            if idx >= 5:
                male_template, female_template = temp
                male_sent = male_template.format(key)
                female_sent = female_template.format(value)
                if key not in special_key_list:
                    male_sent = "My" + male_sent[3:]
                    female_sent = "My" + female_sent[3:]

                result_list = [key, value, male_sent, female_sent]
                output_list.append("\t".join(result_list))
            else:
                template = temp[1] if key in special_key_list else temp[0]
                result_list = [key, value, template.format(key), template.format(value)]
                output_list.append("\t".join(result_list))
    return output_list


def generate_nationality_adj(templates, nation_dict):
    templates = [temp.split("\t") for temp in templates]
    output_list = []
    for idx, arr in enumerate(templates):
        adj_template, nation_template = arr
        for nation, nation_adj in nation_dict.items():
            nation = string.capwords(nation)
            nation_adj = string.capwords(nation_adj)
            result_list = [nation, nation_adj, nation_template.format(nation), adj_template.format(nation_adj)]
            output_list.append("\t".join(result_list))

    return output_list


def generate_datasets(root_dir, category_name, generation_algorithm):
    templates_dir = os.path.join(root_dir, "templates")
    dict_dir = os.path.join(root_dir, "dict")
    template_path = os.path.join(templates_dir, "{0}_templates.txt".format(category_name))
    dict_path = os.path.join(dict_dir, "{0}_words.txt".format(category_name))
    word_dict = {key: value for key, value in read_file(dict_path, preprocess=lambda x: x.strip().split("\t"))}
    templates = list(read_file(template_path, preprocess=lambda x: x.strip()))
    return generation_algorithm(templates, word_dict)


def extract_dict(file_path):
    """
    :type file_path: str
    :param file_path: path of input file
    :return: Dict[str, str]
    """
    file_iter = read_file(file_path, preprocess=lambda x: x.strip().split("\t")[:2])
    return {key.lower(): value.lower() for key, value in file_iter}


def negate_sentence(original_sentence, parser):
    doc = parser(original_sentence)
    for token in doc:
        if token.dep_ == "nsubj":
            head_word = token.head
            if head_word.tag_ == "VBZ":
                negation_word = "does not {0}".format(head_word.lemma_)
            elif head_word.tag_ == "VBP":
                negation_word = "do not {0}".format(head_word.lemma_)
            elif head_word.tag_ == "VBD":
                negation_word = "did not {0}".format(head_word.lemma_)
            else:
                negation_word = "not {0}".format(head_word.text)
            head_position = head_word.idx
            return original_sentence[:head_position] + negation_word + original_sentence[head_position + len(head_word.text):]

    return ""
