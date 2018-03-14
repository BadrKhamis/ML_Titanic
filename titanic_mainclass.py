import pandas as pd
import numpy as np


class titanic:
    def __init__(self,Train_path,Testpath):
        self.__train = pd.read_csv(Train_path)
        self.__test = pd.read_csv(Testpath)
        self.__dataset = pd.combine[self.__train,self.__test]

    def data_analytic(self):
        print(self.__dataset.head(5))
