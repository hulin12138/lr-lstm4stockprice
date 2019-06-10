import sys
import re
import matplotlib.pyplot as plt
import numpy as np
result = []
with open(sys.argv[1],'r') as f:
    for line in f.readlines():
        string = line.strip('\n')
        tmpList = re.split(' ',string)
        result.append([float(n) for n in tmpList])


plt.figure(figsize = (12,4))
result = result[-360:]
for i in range(len(result)):
    if (result[i][1] == -1):
        result[i][1] = result[i][0]
result = np.array(result)
plt.plot(result[:,1:2],color='r')
plt.plot(result[:,0:1],color='b')
plt.show()
