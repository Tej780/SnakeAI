import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


SS = 20

apple = (8,12)

array = np.ones((20, 20))

for i in range(len(array)):
    for j in range(len(array[0])):
        v1 = np.sqrt((apple[0] - i)**2 + (apple[1] - j)**2)
        v2 = (np.abs((i-10) + (j-10)) + np.abs((i-10) - (j-10)))
        v1_normed = v1 #/ np.abs(v1).max(axis=0)
        v2_normed = v2 #/ np.abs(v2).max(axis=0)

        array[i][j] = - (v1_normed + v2_normed)
        print("(", array[i][j], ")", end=" ")
    print("\n")

sns.heatmap(array)
plt.show()


