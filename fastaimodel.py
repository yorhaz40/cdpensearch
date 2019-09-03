
from lang_model_utils import lm_vocab, load_lm_vocab, train_lang_model
from general_utils import save_file_pickle, load_file_pickle
import logging
from pathlib import Path
import torch
import numpy as np
import tqdm

def dataprepare():
    source_path = Path('./data/lang_model/')

    with open(source_path / 'train.docstring', 'r') as f:
        trn_raw = f.readlines()

    with open(source_path / 'valid.docstring', 'r') as f:
        val_raw = f.readlines()

    with open(source_path / 'test.docstring', 'r') as f:
        test_raw = f.readlines()

    vocab = lm_vocab(max_vocab=50000,
                     min_freq=10)

    # fit the transform on the training data, then transform
    trn_flat_idx = vocab.fit_transform_flattened(trn_raw)

    # apply transform to validation data
    val_flat_idx = vocab.transform_flattened(val_raw)

    print(trn_flat_idx[:10])
    print([vocab.itos[x] for x in trn_flat_idx[:10]])


    vocab.save('./data/lang_model/vocab.cls')
    save_file_pickle('./data/lang_model/trn_flat_idx_list.pkl', trn_flat_idx)
    save_file_pickle('./data/lang_model/val_flat_idx_list.pkl', val_flat_idx)


def train():
    vocab = load_lm_vocab('./data/lang_model/vocab.cls')
    trn_flat_idx = load_file_pickle('./data/lang_model/trn_flat_idx_list.pkl')
    val_flat_idx = load_file_pickle('./data/lang_model/val_flat_idx_list.pkl')
    fastai_learner, lang_model = train_lang_model(model_path = './data/lang_model_weights_v2',
                                                  trn_indexed = trn_flat_idx,
                                                  val_indexed = val_flat_idx,
                                                  vocab_size = vocab.vocab_size,
                                                  lr=3e-3,
                                                  em_sz= 500,
                                                  nh= 500,
                                                  bptt=20,
                                                  cycle_len=1,
                                                  n_cycle=3,
                                                  cycle_mult=2,
                                                  bs = 200,
                                                  wd = 1e-6)
    fastai_learner.fit(1e-3, 3, wds=1e-6, cycle_len=2)
    fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=2)
    fastai_learner.fit(1e-3, 2, wds=1e-6, cycle_len=3, cycle_mult=10)
    fastai_learner.save('lang_model_learner_v2.fai')
    lang_model_new = fastai_learner.model.eval()
    torch.save(lang_model_new, './data/lang_model/lang_model_gpu_v2.torch')
from torch.autograd import Variable

def list2arr(l):
    "Convert list into pytorch Variable."

    return Variable(torch.from_numpy(np.expand_dims(np.array(l), -1))).cpu()

def make_prediction_from_list(model, l):
    """
    Encode a list of integers that represent a sequence of tokens.  The
    purpose is to encode a sentence or phrase.

    Parameters
    -----------
    model : fastai language model
    l : list
        list of integers, representing a sequence of tokens that you want to encode

    """
    arr = list2arr(l)# turn list into pytorch Variable with bs=1
    model.reset()  # language model is stateful, so you must reset upon each prediction
    hidden_states = model(arr)[-1][-1] # RNN Hidden Layer output is last output, and only need the last layer

    #return avg-pooling, max-pooling, and last hidden state
    return hidden_states.mean(0), hidden_states.max(0)[0], hidden_states[-1]


def get_embeddings(lm_model, list_list_int):
    """
    Vectorize a list of sequences List[List[int]] using a fast.ai language model.

    Paramters
    ---------
    lm_model : fastai language model
    list_list_int : List[List[int]]
        A list of sequences to encode

    Returns
    -------
    tuple: (avg, mean, last)
        A tuple that returns the average-pooling, max-pooling over time steps as well as the last time step.
    """
    n_rows = len(list_list_int)
    n_dim = lm_model[0].nhid
    avgarr = np.empty((n_rows, n_dim))
    maxarr = np.empty((n_rows, n_dim))
    lastarr = np.empty((n_rows, n_dim))
    count =0
    for i in range(len(list_list_int)):
        count += 1
        avg_, max_, last_ = make_prediction_from_list(lm_model, list_list_int[i])
        avgarr[i,:] = avg_.data.numpy()
        maxarr[i,:] = max_.data.numpy()
        lastarr[i,:] = last_.data.numpy()
        if(count %1000==0):
            print(count)

    return avgarr, maxarr, lastarr

def save_embeddings():
    vocab = load_lm_vocab('./data/lang_model/vocab.cls')
    lang_model = torch.load('./data/lang_model/lang_model_gpu_v2.torch',
                            map_location=lambda storage, loc: storage)
    with open('/home/bohong/文档/to_bro/seq_of_zlb_reverse/data/processed_data/test.function', 'r') as f:
        test_raw = f.readlines()

    idx_docs_test = vocab.transform(test_raw, max_seq_len=30, padding=False)
    avg_hs_test, max_hs_test, last_hs_test = get_embeddings(lang_model, idx_docs_test)
    # save the test set embeddings
    savepath = Path('./data/lang_model_emb/')
    np.save(savepath / 'avg_emb_dim500_code_v2.npy', avg_hs_test)
    np.save(savepath / 'max_emb_dim500_code_v2.npy', max_hs_test)
    np.save(savepath / 'last_emb_dim500_code_v2.npy', last_hs_test)

if __name__ == "__main__":
    #dataprepare()
     train()
   #  save_embeddings()