
import ipdb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from abc import abstractmethod


class Dataset():
    def __init__(self, fold):
        # fold \in {0,1,2,3,4}
        self.fold = fold

    def load_data(self, fname, sep=","):
        df = pd.read_csv(fname, sep=sep)
        df = df.sample(frac=1, random_state=1)
        df.reset_index(inplace=True)
        return df

    def get_feat_types(self, df):
        cat_feat = []
        num_feat = []
        for key in list(df):
            if df[key].dtype == object:
                cat_feat.append(key)
            elif len(set(df[key])) > 2:
                num_feat.append(key)
        return cat_feat, num_feat

    def scale_num_feats(self, df1, df2, num_feat):
        # scale numerical features
        for key in num_feat:
            scaler = StandardScaler()
            df1[key] = scaler.fit_transform(df1[key].values.reshape(-1, 1))
            df2[key] = scaler.transform(df2[key].values.reshape(-1, 1))
        return df1, df2

    def split_data(self, X, y):
        x_chunks = []
        y_chunks = []
        for i in range(5):
            start = int(i / 5 * len(X))
            end = int((i + 1) / 5 * len(X))
            x_chunks.append(X.iloc[start:end])
            y_chunks.append(y.iloc[start:end])

        X_test, y_test = x_chunks.pop(self.fold), y_chunks.pop(self.fold)
        X_train, y_train = pd.concat(x_chunks), pd.concat(y_chunks)

        X_train.reset_index(drop=True, inplace=True)
        y_train.reset_index(drop=True, inplace=True)
        X_test.reset_index(drop=True, inplace=True)
        y_test.reset_index(drop=True, inplace=True)

        return X_train, y_train, X_test, y_test


class CorrectionShift(Dataset):
    def __init__(self, seed):
        super(CorrectionShift, self).__init__(seed)
    def get_mean_var(self):
        num_feat = ["duration", "amount", "age"]
        mean1 = {}
        var1 = {}
        mean2 = {}
        var2 = {}
        for key in num_feat:
            scaler1 = StandardScaler().fit(self.df1[key].values.reshape(-1, 1))
            scaler2 = StandardScaler().fit(self.df2[key].values.reshape(-1, 1))

            mean1[key] = scaler1.mean_
            var1[key] = scaler1.var_
            mean2[key] = scaler2.mean_
            var2[key] = scaler2.var_
        return mean1, var1, mean2, var2
    def get_data(self, fname1, fname2):
        df1 = self.load_data(fname1)
        df2 = self.load_data(fname2)

        num_feat = ["duration", "amount", "age"]
        cat_feat = ["personal_status_sex"]
        target = "credit_risk"
        for feature in cat_feat:

            df1[f'{feature}'].replace({1: 0, 2: 1, 3: 2, 5: 3},inplace=True)
            df2[f'{feature}'].replace({1: 0, 2: 1, 3: 2, 5: 3},inplace=True)

        df1 = df1.drop(columns=[c for c in list(df1) if c not in num_feat + cat_feat + [target]])
        df2 = df2.drop(columns=[c for c in list(df2) if c not in num_feat + cat_feat + [target]])
        self.df1 = df1
        self.df2 = df2


        X1, y1 = df1.drop(columns=[target]), df1[target]
        X2, y2 = df2.drop(columns=[target]), df2[target]

        data1 = self.split_data(X1, y1)
        data2 = self.split_data(X2, y2)

        return data1, data2


class TemporalShift(Dataset):
    def __init__(self, seed):
        super(TemporalShift, self).__init__(seed)

    def get_mean_var(self):
        all_feat = ['Zip', 'NAICS', 'ApprovalDate', 'ApprovalFY', 'Term', 'NoEmp', 'NewExist', 'CreateJob',
						'RetainedJob', 'FranchiseCode', 'UrbanRural','RevLineCr', 'ChgOffDate', 'DisbursementDate',
						'DisbursementGross', 'ChgOffPrinGr', 'GrAppv', 'SBA_Appv', 'New', 'RealEstate', 'Portion',
						'Recession', 'daysterm', 'xx']
        mean1 = {}
        var1 = {}
        mean2 = {}
        var2 = {}
        for key in all_feat:
            scaler1 = StandardScaler().fit(self.df1[key].values.reshape(-1, 1))
            scaler2 = StandardScaler().fit(self.df2[key].values.reshape(-1, 1))

            mean1[key] = scaler1.mean_
            var1[key] = scaler1.var_
            mean2[key] = scaler2.mean_
            var2[key] = scaler2.var_
        return mean1, var1, mean2, var2

    def get_data(self, fname):
        df = self.load_data(fname)
        df = df.fillna(-1)

        # Define target variable
        df["NoDefault"] = 1 - df["Default"].values


        df = df.drop(columns=["Selected", "State", "Name", "BalanceGross", "LowDoc", "BankState",
                              "LoanNr_ChkDgt", "MIS_Status", "Default", "Bank", "City"])

        cat_feat, num_feat = self.get_feat_types(df)
        for feature in cat_feat:
            df[f'{feature}'].replace({'N': 0, 'Y': 1, 'T': 2, '0': 3, -1: 4},inplace=True)

        # Get df1 and df2
        df1 = df[df["ApprovalFY"] < 2007]
        df1 = df1[df1["ApprovalFY"] > 1998]

        df2 = df[df["ApprovalFY"] < 2013]
        df2 = df2[df2["ApprovalFY"] > 1985]


        self.df1 = df1
        self.df2 = df2
        X1, y1 = df1.drop(columns=["NoDefault",'index']), df1["NoDefault"]
        X2, y2 = df2.drop(columns=["NoDefault",'index']), df2["NoDefault"]

        data1 = self.split_data(X1, y1)
        data2 = self.split_data(X2, y2)

        return data1, data2


class GeospatialShift(Dataset):
    def __init__(self, seed):
        super(GeospatialShift, self).__init__(seed)
    def get_mean_var(self):
        all_feat = ['sex', 'age', 'address', 'famsize', 'Pstatus', 'Medu', 'Fedu', 'Mjob',
						   'Fjob', 'reason', 'guardian', 'traveltime', 'studytime', 'failures',
						   'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 'higher',
						   'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc',
						   'health', 'absences']
        mean1 = {}
        var1 = {}
        mean2 = {}
        var2 = {}
        for key in all_feat:
            scaler1 = StandardScaler().fit(self.df1[key].values.reshape(-1, 1))
            scaler2 = StandardScaler().fit(self.df2[key].values.reshape(-1, 1))

            mean1[key] = scaler1.mean_
            var1[key] = scaler1.var_
            mean2[key] = scaler2.mean_
            var2[key] = scaler2.var_
        return mean1, var1, mean2, var2
    def get_data(self, fname, sep):
        df = self.load_data(fname, sep)

        # Define target variable
        df["Outcome"] = (df["G3"] < 10).astype(int)

        # Drop variables highly correlated with target
        df = df.drop(columns=["G1", "G2", "G3"])

        cat_feat, num_feat = self.get_feat_types(df)


        df.replace(
            {
                'school': {'GP': 0, 'MS': 1},
                'sex': {'F': 0, 'M': 1},
                'address': {'U': 0, 'R': 1},
                'famsize': {'LE3': 0, 'GT3': 1},
                'Pstatus': {'T': 0, 'A': 1},
                'Mjob': {'health': 0, 'services': 1, 'other': 2, 'at_home': 3, 'teacher': 4},
                'Fjob': {'health': 0, 'services': 1, 'other': 2, 'at_home': 3, 'teacher': 4},
                'reason': {'home': 0, 'other': 1, 'course': 2, 'reputation': 3},
                'guardian': {'mother': 0, 'other': 1, 'father': 2},
                'schoolsup': {'yes': 0, 'no': 1},
                'famsup': {'yes': 0, 'no': 1},
                'paid': {'yes': 0, 'no': 1},
                'activities': {'yes': 0, 'no': 1},
                'nursery': {'yes': 0, 'no': 1},
                'higher': {'yes': 0, 'no': 1},
                'internet': {'yes': 0, 'no': 1},
                'romantic': {'yes': 0, 'no': 1},

            },
            inplace=True)



        df1 = df[df["school"] == 0]
        df2 = df
        self.df1 = df1
        self.df2 = df2

        df1["Outcome"] = 1 - df1["Outcome"].values
        df2["Outcome"] = 1 - df2["Outcome"].values

        X1, y1 = df1.drop(columns=["Outcome", "school", 'index']), df1["Outcome"]
        X2, y2 = df2.drop(columns=["Outcome", "school", 'index']), df2["Outcome"]

        data1 = self.split_data(X1, y1)
        data2 = self.split_data(X2, y2)

        return data1, data2




class SimulatedData(Dataset):
    def __init__(self, seed):
        self.c0_means = -2 * np.ones(2)
        self.c1_means = 2 * np.ones(2)
        self.c0_cov = 0.5 * np.eye(2)
        self.c1_cov = 0.5 * np.eye(2)
        super(SimulatedData, self).__init__(seed)

    def get_data(self, num_samples=1000):
        np.random.seed(1)

        X0 = np.random.multivariate_normal(self.c0_means, self.c0_cov, int(num_samples / 2))

        X1 = np.random.multivariate_normal(self.c1_means, self.c1_cov, int(num_samples / 2))
        X = np.vstack(np.array([X0, X1]))

        y = np.array([0] * int(num_samples / 2) + [1] * int(num_samples / 2))

        indices = np.arange(num_samples)
        np.random.shuffle(indices)
        X = X[indices]
        y = y[indices]
        data = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "y": y})
        X, y = data.drop(columns=["y"]), data["y"]

        X_train, y_train, X_test, y_test = self.split_data(X, y)

        return X_train, y_train, X_test, y_test

