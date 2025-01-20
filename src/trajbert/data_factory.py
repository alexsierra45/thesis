import math
import torch.utils.data as Data
import torch
from random import *
import os
import pandas as pd

from dataset import DataSet
from utils import make_coocurrence_matrix

class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, masked_tokens, masked_pos, user_ids, day_ids, 
                 input_prior, input_next, input_prior_dis, input_next_dis):
        self.input_ids = input_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.user_ids = user_ids
        self.day_ids = day_ids
        self.input_prior = input_prior
        self.input_next = input_next
        self.input_prior_dis = input_prior_dis
        self.input_next_dis = input_next_dis

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, index):
        return (self.input_ids[index], self.masked_tokens[index], self.masked_pos[index],
                self.user_ids[index], self.day_ids[index], self.input_prior[index],
                self.input_next[index], self.input_prior_dis[index], self.input_next_dis[index])

def data_preprocess(args):
    device = 'cuda:%s' % str(args.gpu)

    if args.data_type in ['cdr', 'tdrive', 'etecsa', 'humob']:
        train_data_name = f'{args.data_type}/train.h5'
        if args.is_training == 1:
            test_data_name = f'{args.data_type}/{'test' if args.data_type in ['cdr', 'tdrive'] 
                                                 else 'valid'}_{args.pre_len}.h5'
        else:
            test_data_name = f'{args.data_type}/test_{args.pre_len}.h5'
            # test_data_name = f'{args.data_type}/hidden/test_{args.pre_len}_{int(args.mask_perc * 100)}_{int(args.hid_perc * 100)}.h5'
    else:
        raise Exception('please check data type', args.data_type)
    
    print('success load ', train_data_name, test_data_name)
    train_df = pd.read_hdf(os.path.join(args.root_path, args.data_path, train_data_name))
    test_df = pd.read_hdf(os.path.join(args.root_path, args.data_path, test_data_name))

    dataset = DataSet(train_df, test_df)
    
    train_data = dataset.gen_train_data()  # [seq, user_index, day]
    test_data = dataset.gen_test_data()  # [seq, masked_pos, masked_tokens, user_index, day]

    train_word_list = list(
        set(str(train_data[i][0][j]) for i in range(len(train_data)) for j in range(len(train_data[i][0]))))
    test_word_list = list(
        set(str(test_data[i][0][j]) for i in range(len(test_data)) for j in range(len(test_data[i][0]))))
    test_masked_list = list(
        set(str(test_data[i][2][j]) for i in range(len(test_data)) for j in range(len(test_data[i][2]))))
    train_word_list.remove('[PAD]')
    test_word_list.remove('[PAD]')

    try:
        test_word_list.remove('[MASK]')
    except:
        pass
    
    train_word_list.extend(test_word_list)
    train_word_list.extend(test_masked_list)
    train_word_list = list(set(train_word_list))
    train_word_list.sort()

    train_word_list_int = [int(train_word_list[i]) for i in range(len(train_word_list))]
    train_word_list_int.sort()

    word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
    for i, w in enumerate(train_word_list_int):
        if w == '[PAD]' or w == '[MASK]':
            print("error")
        word2idx[str(w)] = i + 4

    vocab_size = len(word2idx)

    train_token_list = list()
    train_user_list = list()
    train_day_list = list()
    max_value = 0
    for sentence in train_data:
        seq, user_index, day = sentence
        for s in seq:
            try:
                max_value = max(max_value, word2idx[str(s)])
            except:
                print(s)
                
        arr = [word2idx[s] for s in seq]
        train_token_list.append(arr)
        train_user_list.append(user_index)
        train_day_list.append(day)

    coocurrence_map = make_coocurrence_matrix(token_list=train_token_list, token_size=vocab_size)
    coocurrence_map = torch.Tensor(coocurrence_map).to(device)
    print('vocab size: ', vocab_size)

    return vocab_size, coocurrence_map, train_user_list, train_day_list, train_token_list, word2idx, test_data

def make_train_data(token_list, word2idx, max_pred):
    total_data = []
    vocab_size = len(word2idx)
    for i in range(len(token_list)):
        tokens_a = token_list[i]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']]

        # MASK LM
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                        if token != word2idx['[CLS]'] and token != word2idx['[SEP]'] and token != word2idx[
                            '[PAD]']]  # candidate masked position
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []

        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:  # 80%
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10%
                index = randint(0, vocab_size - 1)  # random index in vocabulary
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD'
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index  # replace


        total_data.append([input_ids, masked_tokens, masked_pos])
    return total_data

def make_test_data(test_data, word2idx):
    # [seq, masked_pos, masked_tokens, user_index, day]
    total_test_data = []
    for sentence in test_data:
        arr = [word2idx[s] for s in sentence[0]]
        user = sentence[3]
        arr = [word2idx['[CLS]']] + arr + [word2idx['[SEP]']]
        masked_pos = [pos + 1 for pos in sentence[1]]
        masked_tokens = [word2idx[str(s)] for s in sentence[2]]
        day = sentence[4]
        total_test_data.append([arr, masked_tokens, masked_pos, user, day])
    return total_test_data

def get_dis_score(dis):
    return 1 / math.log(1 + dis, 2)

def get_id_pn(input_ids, masked_pos, seq_len):
    input_prior = []
    input_next = []
    input_prior_dis = [] 
    input_next_dis = []
    for i in range(len(input_ids)):
        seq = input_ids[i]
        ids_prior = []
        ids_next = []
        ids_prior_dis = [] 
        ids_next_dis = []

        for pos in masked_pos[i]:
            for j in range(pos - 1, -1, -1):
                if seq[j] != 0 and seq[j] != 3:
                    ids_prior.append(seq[j])
                    ids_prior_dis.append(get_dis_score(abs(pos - j)))
                    break
            for j in range(pos, seq_len):
                if seq[j] != 0 and seq[j] != 3:
                    ids_next.append(seq[j])
                    ids_next_dis.append(get_dis_score(abs(j - pos + 1)))
                    break
                
        input_prior.append(ids_prior)
        input_next.append(ids_next)
        input_prior_dis.append(ids_prior_dis)
        input_next_dis.append(ids_next_dis)

    return input_prior, input_next, input_prior_dis, input_next_dis

class data_provider():
    def __init__(self, args):
        self.args = args
        self.vocab_size, self.coocurrence_map, self.train_user_list, \
        self.train_day_list, self.train_token_list, self.word2idx, self.test_data = data_preprocess(args)
    
    def get_loader(self, flag, args):
        device = 'cuda:%s' % str(args.gpu)

        coocurrence_map_copy = self.coocurrence_map.clone()
        for i in range(len(coocurrence_map_copy)):
            coocurrence_map_copy[i][i] = 0

        if flag == 'train':
            self.total_data = make_train_data(self.train_token_list, self.word2idx, int(self.args.pre_len)) 

            input_ids, masked_tokens, masked_pos = zip(*self.total_data)
            input_prior, input_next, input_prior_dis, input_next_dis = get_id_pn(input_ids, masked_pos, self.args.seq_len)

            # Crear el dataset con tensores
            dataset = MyDataSet(
                [torch.tensor(item).to(device) for item in input_ids],
                [torch.tensor(item).to(device) for item in masked_tokens],
                [torch.tensor(item).to(device) for item in masked_pos],
                torch.LongTensor(self.train_user_list).to(device),
                torch.LongTensor(self.train_day_list).to(device),
                [torch.tensor(item).to(device) for item in input_prior],
                [torch.tensor(item).to(device) for item in input_next],
                [torch.tensor(item).to(device) for item in input_prior_dis],
                [torch.tensor(item).to(device) for item in input_next_dis],
            )

            # Crear DataLoader con el collate_fn personalizado
            loader = Data.DataLoader(dataset, batch_size=self.args.bs, shuffle=True, collate_fn=custom_collate_fn)
            return loader
        
        elif flag == 'test' or flag == 'infer':
            self.test_total_data = make_test_data(self.test_data, self.word2idx)
            
            test_input_ids, test_masked_tokens, test_masked_pos, test_user_ids, test_day_ids = zip(*self.test_total_data)
            test_input_prior, test_input_next, test_input_prior_dis, test_input_next_dis = get_id_pn(test_input_ids, test_masked_pos, self.args.seq_len)

            # Crear el dataset con tensores
            dataset = MyDataSet(
                [torch.tensor(item).to(device) for item in test_input_ids],
                [torch.tensor(item).to(device) for item in test_masked_tokens],
                [torch.tensor(item).to(device) for item in test_masked_pos],
                torch.LongTensor(test_user_ids).to(device),
                torch.LongTensor(test_day_ids).to(device),
                [torch.tensor(item).to(device) for item in test_input_prior],
                [torch.tensor(item).to(device) for item in test_input_next],
                [torch.tensor(item).to(device) for item in test_input_prior_dis],
                [torch.tensor(item).to(device) for item in test_input_next_dis],
            )

            # Crear DataLoader con el collate_fn personalizado
            loader = Data.DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=custom_collate_fn)
            return loader
        
    def get_vocabsize(self):
        return self.vocab_size

    def get_coocurrence_map(self):
        return self.coocurrence_map
    
from torch.nn.utils.rnn import pad_sequence
    
def custom_collate_fn(batch):
    """
    Maneja el relleno dinámico y genera máscaras para las secuencias de longitud variable.
    """
    # Separar los componentes del lote
    input_ids, masked_tokens, masked_pos, user_ids, day_ids, \
    input_prior, input_next, input_prior_dis, input_next_dis = zip(*batch)

    # Rellenar dinámicamente las secuencias
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    masked_tokens = pad_sequence(masked_tokens, batch_first=True, padding_value=0)
    masked_pos = pad_sequence(masked_pos, batch_first=True, padding_value=0)
    input_prior = pad_sequence(input_prior, batch_first=True, padding_value=0)
    input_next = pad_sequence(input_next, batch_first=True, padding_value=0)

    # Rellenar las distancias (que son tensores de punto flotante)
    input_prior_dis = pad_sequence(input_prior_dis, batch_first=True, padding_value=0.0)
    input_next_dis = pad_sequence(input_next_dis, batch_first=True, padding_value=0.0)

    # Las máscaras binarias: 1 para valores válidos, 0 para relleno
    input_mask = (masked_tokens != 0).long()

    # Convertir user_ids y day_ids a tensores
    user_ids = torch.stack(user_ids)
    day_ids = torch.stack(day_ids)

    return (input_ids, masked_tokens, masked_pos, user_ids, day_ids, 
            input_prior, input_next, input_prior_dis, input_next_dis, input_mask)