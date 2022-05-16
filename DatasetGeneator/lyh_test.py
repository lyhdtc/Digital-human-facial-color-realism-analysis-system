import numpy as np

# filename = '/home/lyh/results/GIF_DataSetTest1/params.npy'
# a = np.load(filename, allow_pickle=True).item()
# print(type(a))
# filename2 = '/home/lyh/results/GIF_DataSetTest/params.npy'
# b = np.load(filename2, allow_pickle=True).item()
# a = dict()
# np.save(filename, a)
# a = np.load(filename, allow_pickle=True).item()
# print(a)


a = np.array([[1,2,3],
             [2,3,4]])
b = np.array([[4,5,6],
             [5,6,7]])
ans = np.sqrt(np.sum((a-b)**2))
print(ans)