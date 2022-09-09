from run_ner import readfile

wholeword_path = 'data_five/1/test.txt'

def label_to_number(case):
    tokens, labels = case
    # tokens = ''.join(tokens)
    numbers = [1 if label != 'O' else 0 for label in labels]
    return tokens, numbers

def read_data(name = 'data/data_five/1/test.txt'):
    data = readfile(name)
    data = [label_to_number(case) for case in data]
    return data

def dd():
    path = 'data_five/1/test.txt'
    test = read_test(path)
    words, labels = test[0]
    m = Sector_2022(cuda = True, wholeword = True)
    ids = m.toker.encode(words)
    assert len(ids) == len(words) + 2

