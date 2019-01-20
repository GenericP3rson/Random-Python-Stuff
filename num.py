import sys

stuff = []

# i = 0 
# for line in sys.stdin:
#     stuff.append(line)

stuff = [
    [4, 5], 
    [1, 1],
    [4, 4],
    [4, 3],
    [3, 2],
    [2, 4]
]

stuff = [
    [5, 4],
    [2, 1],
    [4, 2],
    [3, 3],
    [3, 4]
]

# The first coordinate is the size of the graph, the second how many further items there are
zz = 0
# while zz < len(stuff)
if stuff[0][0] % 2 == 0:
    i = 1
    while (i < len(stuff)):
        if (stuff[i][0] % 2 == 0):
            if (stuff[i][1] % 2 == 0):
                print stuff[0][0]/2 * (stuff[i][0]-1) + stuff[i][1]/2 
            else:
                print int(round((stuff[0][0] ** 2) / 2) + stuff[0][0]/2 * stuff[i][0])
        else:
            if (stuff[i][1] % 2 == 1):
                print (stuff[0][0]/2 * (stuff[i][0] - 1)) + (stuff[i][1]+1)/2
            else:
                print int(round((stuff[0][0] ** 2) / 2) + (stuff[0][0]/2 * (stuff[i][0] - 1)) + (stuff[i][1]+1)/2)
        i+=1
else:
    i = 1
    while (i < len(stuff)):
        if (stuff[i][1] % 2 == 0):
            if (stuff[i][0] % 2 == 0):
                print round(stuff[0][0]/2) * (stuff[i][0]-1) + stuff[i][1]/2 
            else:
                print int(round((stuff[0][0] ** 2) / 2) + stuff[0][0]/2 * stuff[i][0])
        else:
            if (stuff[i][0] % 2 == 1):
                print (round(stuff[0][0]/2) * (stuff[i][0] - 1)) + (stuff[i][1]+1)/2
            else:
                print int(round((stuff[0][0] ** 2) / 2) + (stuff[0][0]/2 * (stuff[i][0] - 1)) + (stuff[i][1]+1)/2)
        i+=1
