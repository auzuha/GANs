occurences = {}
for i in range(10):
    occurences[i] = 0;
num = 2020;
nstr = str(num)
for i in nstr:
    if int(i) in occurences:
        occurences[int(i)] += 1;
    

flag = 1
for i in range(len(nstr)):
    if nstr[i] != str(occurences[i]):
        flag = -1
print(flag)
