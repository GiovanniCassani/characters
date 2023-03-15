import os
import re
import spacy
import multiprocessing as mp
from datetime import datetime

filter = None


def _mp_clean_file(args):

    return clean_file(args)


def clean_file(in_path):

    """
    :param in_path:     str, indicating the file to clean
    :return:
    """

    nlp = spacy.load("en_core_web_sm")
    nlp.max_length = 10000000000

    l = []
    with nlp.select_pipes(disable=["lemmatizer", "tok2vec", "tagger", "parser"]):
        nlp.enable_pipe("senter")  ## Helps with better segmenting into sentences
        with open(in_path, 'r') as f_in:
            for line in f_in:
                line = re.sub("<[A-Za-z]>", ".", line)
                doc = nlp(line)
                sentence = []
                for token in doc:
                    if token.is_sent_start:
                        if not sentence:
                            continue
                        else:
                            l.append(sentence)
                            sentence = []

                    if token.is_upper is True:                  # Remove all full-caps words
                        continue
                    elif token.text.lower() in filter:          # Remove all words that are in the list of banned words
                        continue
                    elif token.is_alpha:
                        sentence.append(token.text.lower())

    return l


def make(corpus_dir, filter_list, out_path, threads=32):

    """
    :param corpus_dir:      str, indicating the folder containing all corpus files
    :param filter_list:     iterable of str, containing strings which should be removed from the corpus
    :param out_path:        str, indicating where to write clean sentences
    :param threads:         int, the number of CPU cores to parallelize over
    :return:                list of lists of str, each string being a token and each inner list being a sentence
    """

    global filter
    filter = set(filter_list)

    files = [os.path.join(corpus_dir, f) for f in os.listdir(corpus_dir) if os.path.isfile(os.path.join(corpus_dir, f))]

    print(datetime.now().strftime(
        "%d/%m/%Y %H:%M:%S: Started processing {} coca files...".format(len(files)))
    )
    with open(out_path, 'w') as f_out:
        with mp.Pool(threads) as pool:
            outputs = pool.imap(_mp_clean_file, ((f) for f in files))
            for output in outputs:
                for sentence in output:
                    f_out.write(' '.join(sentence))
                    f_out.write('\n')
    print(datetime.now().strftime("%d/%m/%Y %H:%M:%S: Done."))

    filter = None
