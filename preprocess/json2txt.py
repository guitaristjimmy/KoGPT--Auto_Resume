import os
import json
from tqdm import tqdm
import re
import numpy as np
import kss

def json2txt(json_dir, result_dir):
    for json_file in tqdm(os.listdir(json_dir)):
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf8') as jf:
            data = json.load(jf)
        q = data['question'].replace('\n', ' ')
        q = re.sub(pattern='[\[\](){}<>]', repl='', string=q)
        q = re.sub(pattern='(최소.+자)', repl='', string=q)

        sentences = q + data['answer'].replace('\n', ' ')
        with open(os.path.join(result_dir), 'a', encoding='utf8') as f:
            f.writelines(sentences+'\n')


def make_question_answer_data(json_dir, result_dir):
    result = []
    for json_file in tqdm(os.listdir(json_dir)):
        with open(os.path.join(json_dir, json_file), 'r', encoding='utf8') as jf:
            data = json.load(jf)
        q = data['question'].replace('\n', ' ')
        q = re.sub(pattern='[\[\](){}<>]', repl='', string=q)
        q = re.sub(pattern='(최소.+자)', repl='', string=q)

        a = re.sub(pattern='[\[\]{}()@#$%^&*()-_=+`~\'\"/]', repl='', string=data['answer'].replace('\n', ' '))
        answers = ''
        for a_s in kss.split_sentences(a):
            temp = '<s>' + a_s + '</s>'
            if len(answers+temp) < 1024:
                answers += temp
            else:
                break

        result.append(('<s>'+q+'</s>', a))

    result = np.asarray(result)

    np.save(result_dir, result)


def make_txt_one_file(txt_dir, result_dir):
    txt_files = os.listdir(txt_dir)
    if os.path.isfile(result_dir):
        print('Early File Exist, Press Enter than Delete file')
        input('..')
        os.remove(result_dir)

    for txt_file in tqdm(txt_files):
        with open(os.path.join(txt_dir, txt_file), 'r', encoding='utf8') as f:
            data = f.readlines()

        data.append('\n')

        with open(result_dir, 'a', encoding='utf8') as f:
            f.writelines(data)


if __name__ == '__main__':
    # json2txt(json_dir='C:/Users/K/Desktop/I_SW\Python_Note/gpt-2/DB_temp', result_dir='E:\\DB\\자소서_txt\\train_jobkor.txt')
    make_question_answer_data(json_dir='C:/Users/K/Desktop/I_SW\Python_Note/gpt-2/DB_temp',
                              result_dir='E:\\DB\\자소서_txt\\QA_dataset.npy')
    # make_txt_one_file('E:\\DB\\test', 'E:\\DB\\test_smaple.txt')
