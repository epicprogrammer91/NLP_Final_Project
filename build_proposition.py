import pandas as pd
from tqdm import tqdm

def parse_oie_output(filename):
    with open(filename, "r", encoding="utf8") as fout:
        l = fout.readlines()
    line_dicts = {}
    counter = 0
    for line in l:
        if line.strip('\n') == 'None':
            continue
        line = line.split('\t')
        sent = line[0]
        subdict = {'full_sent': sent}
        for i in range(1, len(line)):
            if line[i][:5] == 'ARG0:':
                subdict['ARG0'] = line[i][5:]
            elif line[i][:2] == 'V:':
                subdict['V'] = line[i][2:]
            elif line[i][:5] == 'ARG1:':
                subdict['ARG1'] = line[i][5:]
            elif line[i][:5] == 'ARG2:':
                subdict['ARG2'] = line[i][5:]
            elif line[i][:5] == 'ARG3:':
                subdict['ARG3'] = line[i][5:]
            elif line[i][:5] == 'ARG3:':
                subdict['ARG4'] = line[i][5:]
            elif i == len(line) - 1:
                subdict['conf'] = float(line[i].strip('\n'))
        try:
            line_dicts[line[0]].append(subdict)
        except KeyError:
            line_dicts[line[0]] = []
            line_dicts[line[0]].append(subdict)
        counter += 1
        if counter % 10000 == 0:
            print(f'{counter} subdicts created')
    return line_dicts


def create_propositions(line_dicts, bmap):
    propositions = []
    counter = 0
    book_map = pd.read_csv(bmap, index_col=0)
    book_map.set_index('string_sent', inplace=True)
    for v in line_dicts.values():
        proposition = ''
        string_sent = "".join(v[0]['full_sent'].split(' ')).strip()
        try:
            match = book_map.loc[string_sent].values
            sent, title, author = match[0], match[1], match[2]
        except IndexError:
            if len(v) > 1:
                try:
                    string_sent = "".join(v[1]['full_sent'].split(' ')).strip()
                    match = book_map.loc[string_sent].values
                    sent, title, author = match[0], match[1], match[2]
                except IndexError:
                    continue
            else:
                continue
        except KeyError:
            print(string_sent,v[0]['full_sent'])
            continue
        for subdict in v:
            if 'ARG0' in subdict.keys():
                if proposition == '':
                    proposition += f'{subdict["ARG0"]}'
                else:
                    proposition += f' {subdict["ARG0"]}'
            if 'V' in subdict.keys():
                proposition += f' {subdict["V"]}'
            if 'ARG1' in subdict.keys():
                proposition += f' {subdict["ARG1"]}'
            if 'ARG2' in subdict.keys():
                proposition += f' {subdict["ARG2"]}'
            if 'ARG3' in subdict.keys():
                proposition += f' {subdict["ARG3"]}'
            if 'ARG4' in subdict.keys():
                proposition += f' {subdict["ARG4"]}'
        propositions.append((proposition, sent, title, author))
        counter += 1
        if counter % 10000 == 0:
            print(f'{counter} propositions created')
    df = pd.DataFrame(propositions, columns=['proposition', 'sentence', 'title', 'author'])
    df.to_csv('propositions.csv')



if __name__ == "__main__":
    line_dicts = parse_oie_output('result_books.txt')
    create_propositions(line_dicts, 'sentence_info.csv')
