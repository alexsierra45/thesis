import pandas

class DataSet:
    def __init__(self, train_df, test_df):
        self.train_df = train_df
        self.test_df = test_df

    def gen_train_data(self):
        # ['trajectory', 'user_index', 'day']
        records = []
        for _, row in self.train_df.iterrows():
            seq, user_index, day = row['trajectory'], row['user_index'], row['day']
            records.append([seq, user_index, day])
        print("All train length is " + str(len(records)))
        return records

    def gen_test_data(self):
        # ['trajectory', 'masked_pos', 'masked_tokens']
        test_df = self.test_df
        records = []
        for _, row in test_df.iterrows():
            seq, masked_pos, masked_tokens = row['trajectory'], row['masked_pos'], row['masked_tokens']
            user_index, day = row['user_index'], row['day']
            try:  
                eval(seq)
                seq = eval(seq)
            except:pass
            try:
                seq, masked_pos, masked_tokens = list(seq.split()), list(map(int, masked_pos.split())), \
                                                list(map(int, masked_tokens.split()))
            except:
                seq, masked_pos, masked_tokens = list(seq), list(map(int, masked_pos)), \
                                             list(map(int, masked_tokens))
            records.append([seq, masked_pos, masked_tokens, user_index, day])
        print("All test length is " + str(len(records)))
        return records