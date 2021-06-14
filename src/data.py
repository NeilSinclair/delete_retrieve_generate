"""Data utilities."""
import os
import random
import numpy as np
from nltk import ngrams
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import re

import torch
from torch.autograd import Variable

from src.cuda import CUDA


class CorpusSearcher(object):
    def __init__(self, query_corpus, key_corpus, value_corpus, vectorizer, make_binary=True):
        self.vectorizer = vectorizer
        self.vectorizer.fit(key_corpus)

        self.query_corpus = query_corpus
        self.key_corpus = key_corpus
        self.value_corpus = value_corpus
        
        # rows = docs, cols = features
        self.key_corpus_matrix = self.vectorizer.transform(key_corpus)
        if make_binary:
            # make binary
            self.key_corpus_matrix = (self.key_corpus_matrix != 0).astype(int)

        
    def most_similar(self, key_idx, n=10):
        """ score the query against the keys and take the corresponding values """

        query = self.query_corpus[key_idx]
        query_vec = self.vectorizer.transform([query])
        # print(f'DEBUG: len(query) {len(query)}')
        scores = np.dot(self.key_corpus_matrix, query_vec.T)
        scores = np.squeeze(scores.toarray())
        # print(f'DEBUG: scores {scores}')
        # print(f'scores.size: {scores.size}')
        scores_indices = zip(scores, range(len(scores)))
        # Get the top n scores
        selected = sorted(scores_indices, reverse=True)[:n]

        # use the retrieved i to pick examples from the VALUE corpus
        # create a tuple of the ranked attributed in the document, Value_corpus is the corpus of attributes (other
        # than the current one), i is the index of the score (relative to all the text in the document) and score is
        # the similarity score
        selected = [
            #(self.query_corpus[i], self.key_corpus[i], self.value_corpus[i], i, score) # useful for debugging 
            (self.value_corpus[i], i, score) 
            for (score, i) in selected
        ]

        return selected


def build_vocab_maps(vocab_file):
    assert os.path.exists(vocab_file), "The vocab file %s does not exist" % vocab_file
    unk = '<unk>'
    pad = '<pad>'
    sos = '<s>'
    eos = '</s>'

    lines = [x.strip() for x in open(vocab_file)]

    assert lines[0] == unk and lines[1] == pad and lines[2] == sos and lines[3] == eos, \
        "The first words in %s are not %s, %s, %s, %s" % (vocab_file, unk, pad, sos, eos)

    tok_to_id = {}
    id_to_tok = {}
    for i, vi in enumerate(lines):
        tok_to_id[vi] = i
        id_to_tok[i] = vi

    # Extra vocab item for empty attribute lines
    empty_tok_idx =  len(id_to_tok)
    tok_to_id['<empty>'] = empty_tok_idx
    id_to_tok[empty_tok_idx] = '<empty>'

    return tok_to_id, id_to_tok


def extract_attributes(line, attribute_vocab, use_ngrams=False):
    if use_ngrams:
        # generate all ngrams for the sentence
        grams = []
        for i in range(1, 5):
            try:
                i_grams = [
                    " ".join(gram)
                    for gram in ngrams(line, i) 
                ]
                grams.extend(i_grams)
            except RuntimeError:
                continue

        # filter ngrams by whether they appear in the attribute_vocab
        candidate_markers = [
            (gram, attribute_vocab[gram])
            for gram in grams if gram in attribute_vocab
        ]

        # sort attribute markers by score and prepare for deletion
        content = " ".join(line)
        candidate_markers.sort(key=lambda x: x[1], reverse=True)

        candidate_markers = [marker for (marker, score) in candidate_markers]
        # delete based on highest score first
        attribute_markers = []
        for marker in candidate_markers:
            if marker in content:
                attribute_markers.append(marker)
                content = content.replace(marker, "")
        content = content.split()
    ### Change this so that we take the whole content sentence and then split it and then we take
    ### the whole attribute sentences and do whatveer we need to do (Might be different for the above)    
    else:
        content = []
        attribute_markers = []
        for tok in line:
            if tok in attribute_vocab:
                attribute_markers.append(tok)
            else:
                content.append(tok)

    return line, content, attribute_markers


def read_nmt_data(src, config, tgt, attribute_vocab, train_src=None, train_tgt=None,
        ngram_attributes=False):
    ### --- need to fix up this n-gram attributed --- ###
    # It looks like what's happening here is that this creates a dictionary of 
    # pre_attr["doesn't do"] = pre_salience and pre_attr["doesn't do"] = post_salience
    # where "doesn't do" is an example ngram
    # If it's not in ngram mode, then the pre-attr = post-attr = the vocab word
    # The salience is used to delete the ngrams, starting with the most salient, however, because the 
    # sentences from the rationales method are matched with their style words, I'm going to abandon this method
    # Get the items from the pandas dataframe
    src = pd.read_csv(src)
    src = src.dropna(axis=0)
    # src_lines, src_content, src_attribute = list(zip([src.Target.values, src.Masked.values, 
    #     src.Masked_Words.values]))
    src_lines, src_content, src_attribute = [[t_.split() for t_ in src.Target.values],
                                            [t_.split() for t_ in src.Masked.values], 
                                            [t_.split() for t_ in src.Masked_Words.values]]

    src_tok2id, src_id2tok = build_vocab_maps(config['data']['src_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    # we never need to do the TFIDF thing with the source because 
    # test time is strictly in the src => tgt direction. 
    # But we still both src and tgt dist measurers because training is bidirectional
    #  (i.e., we're autoencoding src and tgt sentences during training)
    src_dist_measurer = CorpusSearcher(
        query_corpus=[' '.join(x) for x in src_attribute],
        key_corpus=[' '.join(x) for x in src_attribute],
        value_corpus=[' '.join(x) for x in src_attribute],
        vectorizer=CountVectorizer(vocabulary=src_tok2id),
        make_binary=True
    )
    src = {
        'data': src_lines, 'content': src_content, 'attribute': src_attribute,
        'tok2id': src_tok2id, 'id2tok': src_id2tok, 'dist_measurer': src_dist_measurer
    }


    tgt = pd.read_csv(tgt)
    tgt = tgt.dropna(axis=0)

    tgt_lines, tgt_content, tgt_attribute = [[t_.split() for t_ in tgt.Target.values],
                                            [t_.split() for t_ in tgt.Masked.values], 
                                            [t_.split() for t_ in tgt.Masked_Words.values]]
    tgt_tok2id, tgt_id2tok = build_vocab_maps(config['data']['tgt_vocab'])
    # train time: just pick attributes that are close to the current (using word distance)
    # because this is only used to noise the inputs
    if train_src is None or train_tgt is None:
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in tgt_attribute],
            key_corpus=[' '.join(x) for x in tgt_attribute],
            value_corpus=[' '.join(x) for x in tgt_attribute],
            vectorizer=CountVectorizer(vocabulary=tgt_tok2id),
            make_binary=True
        )
    # at test time, scan through train content (using tfidf) and retrieve corresponding attributes
    else:
        tgt_dist_measurer = CorpusSearcher(
            query_corpus=[' '.join(x) for x in src_content],
            key_corpus=[' '.join(x) for x in train_tgt['content']],
            value_corpus=[' '.join(x) for x in train_tgt['attribute']],
            vectorizer=TfidfVectorizer(vocabulary=tgt_tok2id),
            make_binary=False
        )
    tgt = {
        'data': tgt_lines, 'content': tgt_content, 'attribute': tgt_attribute,
        'tok2id': tgt_tok2id, 'id2tok': tgt_id2tok, 'dist_measurer': tgt_dist_measurer
    }
    return src, tgt

def sample_replace(lines, dist_measurer, sample_rate, corpus_idx, gen_sim_matrix=False, use_sim_matrix=False, 
                    sim_lookup_table=None):
    """
    replace sample_rate * batch_size lines with nearby examples (according to dist_measurer)
    not exactly the same as the paper (words shared instead of jaccaurd during train) but same idea
    method: pass in a batch of lines, then calculate the similarity of each of those lines to every other
        set of attributes in the document
    """
    out = [None for _ in range(len(lines))]
    for i, line in enumerate(lines):
        original_line = line
        if random.random() < sample_rate:
            if use_sim_matrix:
                # lookup_val = "['<s>','" + str(line) + "','</s>']"
                lookup_val = str(line)
                line = re.sub(r"\[|\]|'", "", sim_lookup_table.loc[lookup_val, "most_similar"]).split(', ')
            else:
                # top match is the current line
                sims = dist_measurer.most_similar(corpus_idx + i)[1:]
                
                try:
                    # here we are changing the line (perturbing it) for that random effect
                    # So, we look for the closest set of attributes that don't have the same attributes
                    # as the current one, and if they don't, we set that to be the set of attributes for the
                    # sentence
                    line = next( (
                        tgt_attr.split() for tgt_attr, _, _ in sims
                        if set(tgt_attr.split()) != set(line[1:-1]) # and tgt_attr != ''   # TODO -- exclude blanks?
                    ) )
                # all the matches are blanks
                except StopIteration:
                    line = []
                line = ['<s>'] + line + ['</s>']

        # corner case: special tok for empty sequences (just start/end tok)
        if len(line) == 2:
            line.insert(1, '<empty>')
        line_closest_pair = str(original_line) + '\t' + str(line) + '\n'
        out[i] = line
    # If we're generating the similarity matrix, write out to file
    if gen_sim_matrix:
        if not os.path.exists('working_dir/'):
            os.makedirs('working_dir/')
        with open('working_dir/similarity_matrix.txt', 'a') as f:
            f.write(line_closest_pair)
    return out


def get_minibatch(lines, tok2id, index, batch_size, max_len, sort=False, idx=None,
        dist_measurer=None, sample_rate=0.0, gen_sim_matrix=False, use_sim_matrix=False,
        sim_lookup_table=None):
    """Prepare minibatch."""
    # FORCE NO SORTING because we care about the order of outputs
    #   to compare across systems
    lines = [
        ['<s>'] + line[:max_len] + ['</s>']
        for line in lines[index:index + batch_size]
    ]

    if dist_measurer is not None:
        lines = sample_replace(lines, dist_measurer, sample_rate, index, gen_sim_matrix,
                                use_sim_matrix, sim_lookup_table)

    lens = [len(line) - 1 for line in lines]
    max_len = max(lens)

    unk_id = tok2id['<unk>']
    input_lines = [
        [tok2id.get(w, unk_id) for w in line[:-1]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    output_lines = [
        [tok2id.get(w, unk_id) for w in line[1:]] +
        [tok2id['<pad>']] * (max_len - len(line) + 1)
        for line in lines
    ]

    mask = [
        ([1] * l) + ([0] * (max_len - l))
        for l in lens
    ]

    if sort:
        # sort sequence by descending length
        idx = [x[0] for x in sorted(enumerate(lens), key=lambda x: -x[1])]

    if idx is not None:
        lens = [lens[j] for j in idx]
        input_lines = [input_lines[j] for j in idx]
        output_lines = [output_lines[j] for j in idx]
        mask = [mask[j] for j in idx]

    input_lines = Variable(torch.LongTensor(input_lines))
    output_lines = Variable(torch.LongTensor(output_lines))
    mask = Variable(torch.FloatTensor(mask))

    if CUDA:
        input_lines = input_lines.cuda()
        output_lines = output_lines.cuda()
        mask = mask.cuda()

    return input_lines, output_lines, lens, mask, idx


def minibatch(src, tgt, idx, batch_size, max_len, model_type, is_test=False, gen_sim_matrix=False,
              use_sim_matrix=False, sim_lookup_table=None):
    if not is_test:
        use_src = random.random() < 0.5
        in_dataset = src if use_src else tgt
        out_dataset = in_dataset
        attribute_id = 0 if use_src else 1
    else:
        in_dataset = src
        out_dataset = tgt
        attribute_id = 1

    if model_type == 'delete':
        inputs = get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        # true length could be less than batch_size at edge of data
        batch_len = len(outputs[0])
        attribute_ids = [attribute_id for _ in range(batch_len)]
        attribute_ids = Variable(torch.LongTensor(attribute_ids))
        if CUDA:
            attribute_ids = attribute_ids.cuda()

        attributes = (attribute_ids, None, None, None, None)

    elif model_type == 'delete_retrieve':
        inputs =  get_minibatch(
            in_dataset['content'], in_dataset['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            out_dataset['data'], out_dataset['tok2id'], idx, batch_size, max_len, idx=inputs[-1])

        if is_test:
            # This dist_measurer has sentence attributes for values, so setting 
            # the sample rate to 1 means the output is always replaced with an
            # attribute. So we're still getting attributes even though
            # the method is being fed content. 
            attributes =  get_minibatch(
                in_dataset['content'], out_dataset['tok2id'], idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dataset['dist_measurer'], sample_rate=1.0)
        else:
            # If we're generating the similarity matrix, then we need to run this for every line
            sample_rate = 1 if gen_sim_matrix else 0.1
            attributes =  get_minibatch(
                out_dataset['attribute'], out_dataset['tok2id'], idx, 
                batch_size, max_len, idx=inputs[-1],
                dist_measurer=out_dataset['dist_measurer'], sample_rate=sample_rate,
                gen_sim_matrix=gen_sim_matrix, use_sim_matrix=use_sim_matrix,
                sim_lookup_table=sim_lookup_table)

    elif model_type == 'seq2seq':
        # ignore the in/out dataset stuff
        inputs = get_minibatch(
            src['data'], src['tok2id'], idx, batch_size, max_len, sort=True)
        outputs = get_minibatch(
            tgt['data'], tgt['tok2id'], idx, batch_size, max_len, idx=inputs[-1])
        attributes = (None, None, None, None, None)

    else:
        raise Exception('Unsupported model_type: %s' % model_type)

    return inputs, attributes, outputs


def unsort(arr, idx):
    """unsort a list given idx: a list of each element's 'origin' index pre-sorting
    """
    unsorted_arr = arr[:]
    for i, origin in enumerate(idx):
        unsorted_arr[origin] = arr[i]
    return unsorted_arr



