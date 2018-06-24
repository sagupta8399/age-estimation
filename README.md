# age-estimation
import tensorflow as tf
import numpy as np
import pandas as pd
import scipy.io as sio


data1 = data11['age']
data2 = data21['feat']
data1=data1.reshape(62328)

trainset= 49860;

agetrain = np.array(np.random.randint(low=0, high=62000, size=trainset, dtype='int'))
data3 = []
data4 = []
for i in range(0,trainset):
    data3.append(data2[agetrain[i]])
    data4.append(data1[agetrain[i]])
data3=np.array(data3)
data4=np.array(data4)
data3 = data3.reshape(-1,4096)
data4 = data4.reshape(-1,1)
latent_dim = 500
 

learning_rate = 0.001





input1 = tf.placeholder(tf.float32,[None,4096])
input3 = tf.placeholder(tf.float32,[None,1])

w1 = tf.Variable(tf.random_normal([4096, 500]))

w2 = tf.Variable(tf.random_normal([500, 1]))
latent = tf.matmul(input1,w1)
nl = tf.nn.relu(latent)
input2 = tf.matmul(latent,w2)
output=tf.nn.relu(input2)



loss=tf.reduce_sum(tf.square(output-input3))





def get_batches(batch_size, dataset, name):
        total_batch = int(dataset.shape[0]/batch_size)
        batches = []
        if total_batch == 0:
            return [dataset]
        for i in range(total_batch):
            if i == total_batch-1:
                batches.append(dataset[i*batch_size:, :])
            else:
                batches.append(dataset[i*batch_size:(i+1)*batch_size, :])
        return batches


# In[ ]:





optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
init = tf.global_variables_initializer()
k=0
with tf.Session() as sess:
    sess.run(init)
    for epoch in range(5):
        k=0
        avg_cost = 0
        total_batch=6000
        #atches_feat = get_batches(100, data3, 'train')
        #atches_age = get_batches(100, data4, 'train')
        for i in range(total_batch):
            batch_x1= data3[k:k+100,:]
            batch_x2= data4[k:k+100]
            k+=100
            _ , c = sess.run([optimizer, loss], feed_dict = {input1: batch_x1, input3: batch_x2})
            #print(c)
            avg_cost +=c/6000
        print ("Epoch:", (epoch+1), "cost =", "{:.5f}".format(avg_cost))




