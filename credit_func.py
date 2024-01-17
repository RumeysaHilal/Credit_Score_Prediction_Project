import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import re

def create_dataFrame(credit_data,csv=False):

    # Preprocess
    credit_data["Payment_of_Min_Amount"].replace("NM","No", inplace=True)
    credit_data['Month'] = pd.to_datetime(credit_data['Month'], format='%m').dt.strftime('%B')

    # Create New Feature in another df
    new_cs = pd.DataFrame()

    new_cs["Customer_ID"]=credit_data["Customer_ID"]
    new_cs["Annual_Icome"] = credit_data["Annual_Income"]
    new_cs["Monthly_Inhand_Salary"] = credit_data["Monthly_Inhand_Salary"]
    new_cs["Amount_invested_monthly"] = credit_data["Amount_invested_monthly"]
    
    new_cs['Total_Payment_to_Credit_Limit_Ratio'] = credit_data['Total_EMI_per_month'] / credit_data['Changed_Credit_Limit']

    new_cs['Debt_per_Credit_Card'] = credit_data['Outstanding_Debt'] / credit_data['Num_Credit_Card']
    new_cs.loc[credit_data['Num_Credit_Card'] == 0, 'Debt_per_Credit_Card'] = 0

    new_cs['Credit_Card_Utilization_Density'] = credit_data['Credit_Utilization_Ratio'] / credit_data['Num_Credit_Card']
    new_cs.loc[credit_data['Num_Credit_Card'] == 0, 'Credit_Card_Utilization_Density'] = 0

    new_cs['Daily_Payment_Delay'] = credit_data['Delay_from_due_date'] / 30 

    new_cs['High_Credit_Use'] = (credit_data['Credit_Utilization_Ratio'] >= 0.7).astype(int)

    new_cs['Loan_Type_Count'] = credit_data['Type_of_Loan'].apply(lambda x: len([loan.strip() for loan in re.split(',|and', x)]))

    label_encoder = LabelEncoder()
    new_cs['Credit_Score'] = label_encoder.fit_transform(credit_data['Credit_Score'])

    if csv:
        new_cs.to_csv("new_cs.csv")

    return new_cs

url = "credit-score.csv"
credit_data = pd.read_csv(url)
new_= create_dataFrame(credit_data, csv=True)

