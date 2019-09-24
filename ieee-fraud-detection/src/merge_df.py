import pandas as pd

def mergeAllFile():
    file_trans_name = '../input/ieee-fraud-detection/train_transaction.csv'
    file_id_name = '../input/ieee-fraud-detection/train_identity.csv'
    file_output_name = 'train_all.csv'

    file_trans_name_test = '../input/ieee-fraud-detection/test_transaction.csv'
    file_id_name_test = '../input/ieee-fraud-detection/test_identity.csv'
    file_output_name_test = 'test_all.csv'
    
    mergefile(file_trans_name, file_id_name, file_output_name)
    mergefile(file_trans_name_test, file_id_name_test, file_output_name_test)

def mergefile(file_trans, file_id, file_output):
    trans = pd.read_csv(file_trans)
    id_df = pd.read_csv(file_id)
    trans.merge(id_df, on="TransactionID", how="left")
    trans.to_csv(file_output)

mergefile("../input/train_transaction.csv", "../input/train_identity.csv", "../input/train_all.csv")
mergefile("../input/test_transaction.csv", "../input/test_identity.csv", "../input/test_all.csv")
