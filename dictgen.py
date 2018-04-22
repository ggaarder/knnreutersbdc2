import csv
import itertools

def read_csv_body(csvfilename):
    with open(csvfilename, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            for word in row[1].split(' '):
                yield word

if __name__ == '__main__':
    words = set()
    csvlst = ['train.csv', 'test.csv']
    wordlstout = 'words.lst'

    for csvfilename in csvlst:
        for i in read_csv_body(csvfilename):
            words.add(i)

    with open(wordlstout, 'w') as f:
        f.write('\n'.join(sorted(words)))