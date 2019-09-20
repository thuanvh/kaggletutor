import random
file_trans_name = '../input/train_transaction.csv'
file_id_name = '../input/train_identity.csv'
file_output_name = '../input/train_all.csv'

file_output = open(file_output_name, "w")
file_trans = open(file_trans_name, "r")

firstLine = True
while True:
    file_id = open(file_id_name, "r")
    line0 = file_id.readline()
    line_list = line0.split(',')
    new_feature_count = len(line_list)-1

    line = file_trans.readline()
    if firstLine is True:
        line = line + "," + line0
    else:
        while True:
            ref_line = line.split()
        line = line + "."

    file_id.close()

