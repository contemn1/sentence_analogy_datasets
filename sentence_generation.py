import json
import os
import string

from io_util import read_file, output_list_to_file


def read_template_dict(template_name, category_name):
    root_dir = "/home/zxj/Data/multinli_1.0"
    template_path = os.path.join(root_dir, template_name)
    dict_path = os.path.join(root_dir, "word-pairs-per-category.json")
    pair_dict = json.load(open(dict_path))[category_name]
    templates = read_file(template_path, preprocess=lambda x: x.strip())
    return templates, pair_dict


def generate_capital_countries(template_name, category_name):
    templates, country_dict = read_template_dict(template_name, category_name)
    result_list = []
    for tem in templates:
        for country, city in country_dict.items():
            output_list = [country, city, tem.format(country), tem.format(city)]
            result_list.append("\t".join(output_list))

    return result_list


def generate_currency(root_dir, template_name, category_name):
    templates, currency_dict = read_template_dict(template_name, category_name)
    templates = [temp.split("\t") for temp in templates]
    quadruple_path = os.path.join(root_dir, "currency_word_pairs.txt")
    currency_country_quads = list(
        read_file(quadruple_path, preprocess=lambda x: x.strip().split(" ")))
    one_parameter_templates = templates[:4]
    two_parameter_templates = templates[5:]
    result_list = []
    for idx, template_arr in enumerate(one_parameter_templates):
        currency_template, country_template = template_arr
        for country, currency in currency_dict.items():
            currency = string.capwords(currency) if idx == 0 else currency
            currency_sent = currency_template.format(currency) if idx < 2 else currency_template.format(currency,
                                                                                                        country)
            country_sent = country_template.format(country)
            output_list = [country, currency, country_sent, currency_sent]
            result_list.append("\t".join(output_list))

    for idx, template_arr in enumerate(two_parameter_templates):
        currency_template, country_template = template_arr
        for arr in currency_country_quads:
            country1, currency1, country2, currency2 = arr
            currency1 = string.capwords(currency1) if idx == 0 else currency1
            country_sent = country_template.format(country1, country2)
            currency_sent = currency_template.format(currency1, currency2)
            output_list = [country1, currency1, currency2, currency2, country_sent, currency_sent]
            result_list.append("\t".join(output_list))

    return result_list


def generate_city_in_state(template_name, category_name):
    templates, city_dict = read_template_dict(template_name, category_name)
    result_list = []
    for idx, temp in enumerate(templates):
        for city, state in city_dict.items():
            city = string.capwords(city)
            state = string.capwords(state)
            if idx >= 4:
                city_template, state_template = temp.split("\t")
            else:
                city_template = temp
                state_template = temp

            output_list = [city, state, city_template.format(city), state_template.format(state)]
            result_list.append("\t".format(output_list))

    return result_list


def generate_family(template_name, category_name):
    special_key_list = {"boy", "king", "groom", "prince", "husband", "man"}
    templates, family_dict = read_template_dict(template_name, category_name)
    templates = [temp.split("\t") for temp in templates]
    result_list = []
    for idx, temp in enumerate(templates):
        for key, value in family_dict.items():
            if idx >= 4:
                male_template, female_template = temp
                result_list.append(male_template.format(
                    key) + "\t" + female_template.format(value))
            else:
                template = temp[1] if key in special_key_list else temp[0]
                result_list.append(template.format(
                    key) + "\t" + template.format(value))
    return result_list


def generate_nationality_adj(template_name, category_name):
    templates, family_dict = read_template_dict(template_name, category_name)
    templates = [temp.split("\t") for temp in templates]
    result_list = []
    for idx, arr in enumerate(templates):
        adj_template, nation_template = arr
        for nation, nation_adj in family_dict.items():
            nation = string.capwords(nation)
            nation_adj = string.capwords(nation_adj)
            result_list.append(nation_template.format(
                nation) + "\t" + adj_template.format(nation_adj))

    return result_list


if __name__ == "__main__":
    template_name = "capital_world_templates.txt"
    category_name = "capital-world"
    result_list = generate_capital_countries(template_name, category_name)
    output_path = "/home/zxj/Data/sentence_analogy_datasets/capital_world_pairs.txt"
    output_list_to_file(output_path, result_list)
