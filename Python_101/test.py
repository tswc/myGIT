# # import random as rd
# # import numpy as np
# # a = np.array([[1,2]
# #      ,[2,3]
# #      ,[3,4]]*10)
# # print a.shape[0]
# # rd.shuffle(a)
# # print a
#
# m = 5
# print int(round(5* (1- 0.5)))
# print int(0.5)
# print float(201)/4

# import numpy as np
# import matplotlib.pyplot as plt
# x = np.linspace(0,10,20)
# y = np.linspace(0,10,20)
# i = '111\n 222'
#
# plt.figure(figsize=[10,10])
# plt.subplot(2,3,(1,2))
# plt.plot(x)
# plt.title(i)
#
# plt.figtext(0.7, 0.8,'1213', color='green')
# plt.subplots_adjust(bottom=.01, top=.95, left=.01, right=.99)
# # plt.savefig('testpic.png')
# plt.show()


# i='11'
# b = '22'
# c= i+b
# print c

# import numpy as np
# a = np.array([[1, 2, 3, 4],
#              [2, 3, 4, 5]])
# print np.transpose(a)

# import numpy as np
#
# a = np.array([[1,2,3,4,5,6,7,8,9,10],
#              range(10,20)])
# a = a.T
# mask = (a[:,0]<8)&(a[:,0]>2)
# b = np.where(mask)
# print b
# a = np.delete(a,np.s_[b[0]],axis=0)
# print a

a = range(33,35)
print a