from pathlib import Path
import logging
import wget
import pickle
from typing import List, Callable, Union, Any
from more_itertools import chunked
from itertools import chain
# import nmslib
from pathos.multiprocessing import Pool, cpu_count
from math import ceil


def save_file_pickle(fname:str, obj:Any):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_file_pickle(fname:str):
    with open(fname, 'rb') as f:
        obj = pickle.load(f)
        return obj


def read_training_files(data_path:str):
    """
    Read data from directory
    """
    PATH = Path(data_path)

    with open(PATH/'train.function', 'r',encoding="utf-8") as f:
        t_enc = f.readlines()

    with open(PATH/'valid.function', 'r',encoding="utf-8") as f:
        v_enc = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_enc = t_enc + v_enc

    with open(PATH/'test.function', 'r',encoding="utf-8") as f:
        h_enc = f.readlines()

    with open(PATH/'train.docstring', 'r',encoding="utf-8") as f:
        t_dec = f.readlines()

    with open(PATH/'valid.docstring', 'r',encoding="utf-8") as f:
        v_dec = f.readlines()

    # combine train and validation and let keras split it randomly for you
    tv_dec = t_dec + v_dec

    with open(PATH/'test.docstring', 'r',encoding="utf-8") as f:
        h_dec = f.readlines()


    with open(PATH/'train.api') as f:
        t_api = f.readlines()
    with open(PATH/'valid.api') as f:
        v_api = f.readlines()
    with open(PATH/'test.api') as f:
        h_api = f.readlines()
    tv_api = t_api + v_api


    with open(PATH/'train.seq') as f:
        t_seq = f.readlines()
    with open(PATH/'valid.seq') as f:
        v_seq = f.readlines()
    with open(PATH/'test.seq') as f:
        h_seq = f.readlines()
    tv_seq = t_seq + v_seq



    logging.warning(f'Num rows for encoder training + validation input: {len(tv_enc):,}')
    logging.warning(f'Num rows for encoder holdout input: {len(h_enc):,}')

    logging.warning(f'Num rows for decoder training + validation input: {len(tv_dec):,}')
    logging.warning(f'Num rows for decoder holdout input: {len(h_dec):,}')

    logging.warning(f'Num rows for encoder training + validation input: {len(tv_api):,}')
    logging.warning(f'Num rows for encoder holdout input: {len(h_api):,}')

    logging.warning(f'Num rows for encoder training + validation input: {len(tv_seq):,}')
    logging.warning(f'Num rows for encoder holdout input: {len(h_seq):,}')

    return tv_enc, h_enc, tv_dec, h_dec,tv_api,h_api,tv_seq,h_seq


def apply_parallel(func: Callable,
                   data: List[Any],
                   cpu_cores: int = None) -> List[Any]:
    """
    Apply function to list of elements.
    Automatically determines the chunk size.
    """
    if not cpu_cores:
        cpu_cores = cpu_count()

    try:
        chunk_size = ceil(len(data) / cpu_cores)
        pool = Pool(cpu_cores)
        transformed_data = pool.map(func, chunked(data, chunk_size), chunksize=1)
    finally:
        pool.close()
        pool.join()
        return transformed_data


def flattenlist(listoflists:List[List[Any]]):
    return list(chain.from_iterable(listoflists))


processed_data_filenames = [
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/test_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/train_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.docstring',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/valid_original_function.json.gz',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.function',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings.lineage',
'https://storage.googleapis.com/kubeflow-examples/code_search/data/without_docstrings_original_function.json.gz']


def get_step2_prerequisite_files(output_directory):
    outpath = Path(output_directory)
    assert not list(outpath.glob('*')), f'There are files in {str(outpath.absolute())}, please clear files or specify an empty folder.'
    outpath.mkdir(exist_ok=True)
    print(f'Saving files to {str(outpath.absolute())}')
    for url in processed_data_filenames:
        print(f'downloading {url}')
        wget.download(url, out=str(outpath.absolute()))


def create_nmslib_search_index(numpy_vectors):
    """Create search index using nmslib.

    Parameters
    ==========
    numpy_vectors : numpy.array
        The matrix of vectors

    Returns
    =======
    nmslib object that has index of numpy_vectors
    """

    search_index = nmslib.init(method='hnsw', space='cosinesimil')
    search_index.addDataPointBatch(numpy_vectors)
    search_index.createIndex({'post': 2}, print_progress=True)
    return search_index
