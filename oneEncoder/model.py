from seq2seq_utils import build_seq2seq_model, load_encoder_inputs,load_decoder_inputs,load_text_processor
from pathlib import Path
from general_utils import  read_training_files
from keras.models import Model, load_model
from ktext.preprocess import processor
import dill as dpickle
import numpy as np
OUTPUT_PATH = Path('../../data/seq2seq/')
OUTPUT_PATH.mkdir(exist_ok=True)
from keras.models import Model, load_model
import pandas as pd
import logging
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras import optimizers
import os
from seq2seq_utils import Seq2Seq_Inference


def datapre():
    train_code, holdout_code, train_comment, holdout_comment = read_training_files('./data/processed_data2/')

    assert len(train_code) == len(train_comment)
    assert len(holdout_code) == len(holdout_comment)
    code_proc = processor(heuristic_pct_padding=.7, keep_n=40000)
    print("start")
    t_code = code_proc.fit_transform(train_code)
    print("finish code")
    comment_proc = processor(append_indicators=True, heuristic_pct_padding=.7, keep_n=40000, padding='post')
    t_comment = comment_proc.fit_transform(train_comment)
    print("finish comment")
    with open(OUTPUT_PATH/'py_code_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(code_proc, f)

    with open(OUTPUT_PATH/'py_comment_proc_v2.dpkl', 'wb') as f:
        dpickle.dump(comment_proc, f)

    np.save(OUTPUT_PATH/'py_t_code_vecs_v2.npy', t_code)
    np.save(OUTPUT_PATH/'py_t_comment_vecs_v2.npy', t_comment)


def train():
    encoder_input_data, encoder_seq_len = load_encoder_inputs(OUTPUT_PATH / 'py_t_code_vecs_v2.npy')
    s_encoder_input_data, s_encoder_seq_len = load_encoder_inputs(OUTPUT_PATH/'py_t_seq_vecs_v2.npy')
    decoder_input_data, decoder_target_data = load_decoder_inputs(OUTPUT_PATH / 'py_t_comment_vecs_v2.npy')
    num_encoder_tokens, enc_pp = load_text_processor(OUTPUT_PATH / 'py_code_proc_v2.dpkl')
    s_num_encoder_tokens, s_enc_pp = load_text_processor(OUTPUT_PATH/'py_seq_proc_v2.dpkl')
    num_decoder_tokens, dec_pp = load_text_processor(OUTPUT_PATH / 'py_comment_proc_v2.dpkl')
    seq2seq_Model = build_seq2seq_model(word_emb_dim=128,
                                        hidden_state_dim=128,
                                        encoder_seq_len=encoder_seq_len,
                                        num_encoder_tokens=num_encoder_tokens,
                                        num_decoder_tokens=num_decoder_tokens)


    seq2seq_Model.summary()
    seq2seq_Model.compile(optimizer=optimizers.Nadam(lr=0.0005), loss='sparse_categorical_crossentropy')

    script_name_base = 'py_func_sum_v9_'
    csv_logger = CSVLogger('{:}.log'.format(script_name_base))

    model_checkpoint = ModelCheckpoint('{:}.epoch{{epoch:02d}}-val{{val_loss:.5f}}.hdf5'.format(script_name_base),
                                       save_best_only=True)

    batch_size = 100
    epochs = 50
    history = seq2seq_Model.fit([encoder_input_data, decoder_input_data], np.expand_dims(decoder_target_data, -1),
              batch_size=batch_size,
              epochs=epochs,
              validation_split=0.12, callbacks=[csv_logger, model_checkpoint])
    seq2seq_Model.save("seqmodel.hdf5")


def predict():
    train_code, holdout_code, train_comment, holdout_comment = read_training_files('../../data/processed_data/')
    loc = "/home/bohong/文档/mygit/cdpensearch/cdpensearch/oneEncoder/seqmodel.hdf5"
    seq2seq_Model = load_model(loc)

    loc = OUTPUT_PATH/'py_code_proc_v2.dpkl'
    num_encoder_tokens, enc_pp = load_text_processor(OUTPUT_PATH / 'py_code_proc_v2.dpkl')
    num_decoder_tokens, dec_pp = load_text_processor(OUTPUT_PATH / 'py_comment_proc_v2.dpkl')
    seq2seq_inf = Seq2Seq_Inference(encoder_preprocessor=enc_pp,
                                    decoder_preprocessor=dec_pp,
                                    seq2seq_model=seq2seq_Model)
    demo_testdf = pd.DataFrame({'code': holdout_code, 'comment': holdout_comment, 'ref': ''})
    # seq2seq_inf.predications(df=demo_testdf)
    f = open("generatetag.txt")
    score = seq2seq_inf.evaluate_model(f.readlines(),holdout_comment,max_len=None)
    f.close()
    print(score)
if __name__=="__main__":
    # datapre()
    # train()
    predict()

