import random
files = [
    '../input/train_transaction.csv',
    '../input/train_identity.csv',
    ]
number = 100
maxsize = number * 10
random_lines = [random.randrange(maxsize) for i in range(number)]
print(random_lines)

for fname in files:
    print(fname)
    index = 0
    fin = open(fname, 'r')
    fout = open(fname + str(number) + ".csv", 'w')
    while True:
        print(index)
        line = fin.readline()
        if index == 0 :
            fout.write(line)
        else:
            if index in random_lines:
                fout.write(line)
        index += 1
        if index > maxsize:
            break
        print(fname, index)
    fin.close()
    fout.close()
