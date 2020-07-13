import ndjson
import pickle
import os
from collections import defaultdict
from tqdm import tqdm
import random
from multiprocessing.dummy import Pool as ThreadPool
from multiprocessing import Pool
import threading

def chunks(l,n):
    list_of_lists = []
    for i in range(1, len(l), n):
        list_of_lists.append(l[i:i + n])
    return list_of_lists



path = '/scratche/home/<user>/google_draw/'
path_to_store = '/scratche/home/<user>/processed_quick_draw'
#file = ndjson.load(open(path))

label2key = defaultdict(list)
key2image = {}
key2label = {}
train_keys = []
valid_keys = []
train_path = []
valid_path = []
key2array_idx = {}
array_idx2key = {}
train_path = []
valid_path = []
num_threads = 27

def prepare_data(file_list):

    for i in tqdm(range(len(file_list))):
        f = file_list[i]
        to_store = {}
        if f['recognized'] == True:
            sub_dir = f['word']
            key = f['key_id']
            drawing = f['drawing']
            to_store[key] = drawing
            #if not os.path.exists(os.path.join(path_to_store, sub_dir)):
            try:
                os.mkdir(os.path.join(path_to_store, sub_dir))
            except:
                pass
            p = os.path.join(path_to_store, sub_dir)
            pickle.dump(to_store, open(p+'/'+ str(key)+'.pkl', 'wb'))




for r, d, f in os.walk(path):
    for file in tqdm(f):
        if '.ndjson' in file:
            filee = ndjson.load(open(os.path.join(path,file)))
            chunks_pool = chunks(filee[0:24000], 500)
            pool = ThreadPool(num_threads)
            results = pool.map(prepare_data, chunks_pool)
