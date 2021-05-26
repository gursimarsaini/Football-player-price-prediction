from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd

class Transformer(object):

    def __init__(self, all_cols, cat_cols):
        self.all_cols = all_cols
        self.cat_cols = cat_cols
        self.column_dict = {i:[] for i in self.cat_cols}
        self.scalar = MinMaxScaler()

    def start(self, df, train=False):

        df = df[self.all_cols]

        df = self.fix_fpl_sel(df)
        self.fix_index(df)
        cont_df = df.drop(self.cat_cols,axis=1)
        cont_df = self.log_transform_cont(cont_df)
        if train:
            df = self.one_hot_encode_train(df)
            cont_df_scaled = self.scalar.fit_transform(cont_df)
        
        else:
            df = self.one_hot_encode_test(df)
            cont_df_scaled = self.scalar.transform(cont_df)

        df[cont_df.columns] = cont_df_scaled
        return df

    
    def fix_fpl_sel(self, df):
        df.fpl_sel = df.fpl_sel.apply(lambda x : float(x[:-1]))
        return df
    
    def fix_index(self,df):
        df.reset_index(drop=True, inplace=True)

    def one_hot_encode_train(self, df):
        for i in self.cat_cols:
            temp = pd.get_dummies(df[i], prefix=i)
            self.column_dict[i].extend(temp.columns)
            others = pd.Series(np.zeros(df.shape[0]), name=i+'_others')
            temp = pd.concat([temp, others], axis=1)
            df = pd.concat([df.drop(i, axis=1), temp], axis=1)
        return df
    
    def one_hot_encode_test(self, df):
        for i in self.cat_cols:
            temp = pd.get_dummies(df[i], prefix=i)

            drop1 = set(self.column_dict[i])
            drop2 = set(temp.columns)
            drop_cols = list(drop1.intersection(drop2))

            orig = pd.DataFrame(np.zeros((temp.shape[0], len(self.column_dict[i]))), columns=self.column_dict[i])
            orig[drop_cols] = temp[drop_cols]

            others = temp.drop(drop_cols, axis=1)
            if others.shape[1] == 0:
                others = pd.Series(np.zeros(temp.shape[0]))
            else:
                others = others.apply(np.sum, axis=1)
            others.name = i+'_others'
            df = pd.concat([df.drop(i, axis=1), orig, others], axis=1)
        return df
    
    def log_transform_cont(self, df):
        return df.apply(lambda x : np.log(x+0.01))