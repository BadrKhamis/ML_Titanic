import pandas as pd
import numpy as np


class titanic:
    def __init__(self,Train_path,Testpath):
        self.__train = pd.read_csv(Train_path)
        self.__test = pd.read_csv(Testpath)
        # self.__dataset = pd.combine[self.__train,self.__test]

    def data_analytic(self):
        ''' looking into the data to Analyze, identify patterns, and explore the useful patterns in order
         to build the prediction model '''
        '''print(self.__train.head(5)) ; print(self.__train.tail(5))
        print(self.__train.info())
        print(self.__train.describe())
        print(self.__train.describe(include=['O']))


        '''
        print(self.__train.describe())




if __name__ == '__main__':
    t_path = '/Users/badrkhamis/Documents/ml_experenice/ML_Titanic/dataset/train.csv'
    te_path = '/Users/badrkhamis/Documents/ml_experenice/ML_Titanic/dataset/test.csv'
    titanic_p = titanic(t_path,te_path)
    titanic_p.data_analytic()
