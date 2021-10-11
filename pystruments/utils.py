import csv

import numpy as np


def list_to_comma_separated_string(list_of_values):
    return ','.join([str(x) for x in list_of_values])


def digitise(number):
    """
    This function maps a floating-point number in the range [-1.0,1.0]
    on an integer number in the range [-128,-127,...,127].
    """

    def transform(x):
        return int(x * 127.5 - 0.5)

    number_type = type(number)
    if number_type == list:
        digitised = list()
        for item in number:
            digitised.append(transform(item))
    elif number_type == float:
        digitised = transform(number)
    else:
        raise Exception('Input is neither float nor list-of-float.')
    return digitised


def isdigit(character):
    try:
        float(character)
        return True
    except ValueError:
        return False


def csv_to_list(csv_file):
    if csv_file[-4:] != '.csv':
        raise Exception('String should be in csv format.')
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        output = list(reader)
    return output


def list_to_csv(waveform, file_name):
    with open(file_name + '.csv', 'w') as f:
        writer = csv.writer(f)
        for elem in waveform:
            writer.writerow(elem)


def config_txt_to_dict(txt_file):
    if txt_file[-4:] != '.txt':
        raise Exception('String should be in txt format.')
    my_config = dict()
    with open(txt_file, 'r') as f:
        for line in f:
            if line[0] != '#':
                (key, val) = line.split()
                if isdigit(val):
                    val = float(val)
                    if val == int(val):
                        val = int(val)
                my_config[key] = val
    return my_config


def isbool(v):
    return isinstance(v, bool)


def check_if_every_item_of_list_1_exist_in_list_2(list_1, list_2):
    for item in list_1:
        if item not in list_2:
            raise Exception('Item ', item, 'is not recognised.')


def find_nearest_index(values, target):
    array = np.asarray(values)
    idx = (np.abs(array - target)).argmin()
    return idx


def find_nearest_value(values, target):
    idx = find_nearest_index(values, target)
    return values[idx]
