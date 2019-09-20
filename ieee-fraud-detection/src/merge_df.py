import pandas as pd

file_trans_name = '../input/train_transaction.csv'
file_id_name = '../input/train_identity.csv'
file_output_name = '../input/train_all.csv'

trans = pd.read_csv(file_trans_name)
id_df = pd.read_csv(file_id_name)

trans.merge(id_df, on="TransactionID", how="left")

trans.to_csv(file_output_name)
