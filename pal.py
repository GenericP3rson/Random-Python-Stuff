import sys

stuff = []
i = 0 
for line in sys.stdin:
    stuff.append(line)

# stuff = [5, 6,
# "abccba",
# 2,
# "cf",
# 4,
# "adfa",
# 8,
# "abaazaba",
# 2,
# "ml"]
# lett = []
# i =0
lettst = "abcdefghijklmnopqrstuvwxyz"
def search(y, z):
    i = 0
    while i < len(y):
        if (y[i] == z):
            return i
        i+=1
# while i < 26:
#     lett.append([lettst[i], i])
#     i+=1
# print lett
allo = []
i = 0
while (i < int(stuff[0])):
    x = 0
    arr = []
    while x < len(stuff[2 + 2*i]):
        arr.append(search(lettst, stuff[2+2*i][x]))
        x+=1
    allo.append(arr)
    i+=1
# print allo
i = 0
while i < len(allo):
    ii = 0
    ans = True
    while ii < len(allo[i])/2:
        if (not (allo[i][ii] == allo[i][-1 - ii]) and not (abs(int(allo[i][ii]) - int(allo[i][-1 - ii])) == 2)):
            ans = False
            # print ans, allo[i][ii], allo[i][-1-ii]
        ii+=1
    if (ans):
        sys.stdout.write("YES")
    else:
        sys.stdout.write("NO")
    i+=1