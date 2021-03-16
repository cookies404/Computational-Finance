import numpy as np
import pandas as pd
import math
import scipy.stats as st
import itertools

stock_Ui = np.zeros([18, 127])
Realized_volatility = np.zeros(18)
path = r'C:\Users\hasee\Desktop\CUHK\CMSC5718-Computational Finance\assignment_2_stock_data.xlsx'
sheet = pd.read_excel(path, sheet_name='stock prices')
for i in range(3, 21):
    # get one stock data
    single_stock_data = sheet.iloc[5:133, i]
    #print(single_stock_data)
    for j in range(5, 128):
        stock_Ui[i - 3][j - 5] = math.log(single_stock_data[j + 1] / single_stock_data[j])

for i in range(18):
    each_stock_data = stock_Ui[i, :]
    annualized_standard_deviation = np.std(each_stock_data, ddof=1) * np.sqrt(252)
    Realized_volatility[i] = annualized_standard_deviation
print(
    '\n-------------------------------------------------------q1--------------------------------------------------------------\n')
print("Realized_volatility: ", Realized_volatility)

