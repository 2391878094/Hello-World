
import numpy as np

import random

import matplotlib.pyplot as plt

 

 

population_size = 500  #种群数量

pc = 0.6    #交配概率

pm = 0.01   #变异概率

n_generations = 500   #迭代次数

DNA_length = 15  #染色体长度，一定要大于十进制范围转为二进制的两倍

x_bound = [0, 15]

fig = plt.figure()

 #如果想调大精度，使结果更接近，调大种群数量和迭代次数即可

 
#函数

def F(x):

    return  -3*(x-30) **2*np.sin(x)

 

 

class GA(object):

    #构建50个十五长度染色体（个体）的用0和1表示的基因型

    def __init__(self):

        self.populations = np.random.randint(0, 2, (population_size, DNA_length))

 

    # 对所有DNA进行解码，翻译成10进制数字

    def DNA_decode(self, loser_winner):

        return np.dot(loser_winner, 2 ** np.arange(DNA_length)) / (2 ** DNA_length-1) * x_bound[1]

 
   #计算适应度

    def calculate_fitness(self, loser_winner):

        DNA_value = self.DNA_decode(loser_winner)

        fitness = F(DNA_value)

        fitness = np.where(fitness < 0.0, 0.0, fitness)

        return fitness

 
#选择

    def selection(self):

        # 从0~DNA.length-1中随机选择2组DNA编号

        loser_winner_id = np.random.choice(np.arange(population_size), size=2, replace=False)

        loser_winner = self.populations[loser_winner_id]

        loser_winner_fitness = self.calculate_fitness(loser_winner)

        return loser_winner[np.argsort(loser_winner_fitness)], loser_winner_id

 

    def crossover(self, loser_winner):

        # 必须指定类型为np.bool，否则True会变为1.0000的小数

        crossover_points = np.empty((DNA_length,)).astype(np.bool)

        for i in range(DNA_length):

            crossover_points[i] = True if random.random() < pc else False

        loser_winner[0, crossover_points] = loser_winner[1, crossover_points]

        return loser_winner

 
#突变

    def mutation(self, loser_winner):

        for i in range(DNA_length):

            loser_winner[0, i] = 1 if loser_winner[0, i] == 0 else 0

        return loser_winner

 
#发展，即进行迭代

    def evolve(self):

        for i in range(int(population_size/2)):

            loser_winner, loser_winner_id = self.selection()

            loser_winner = self.crossover(loser_winner)

            loser_winner = self.mutation(loser_winner)

            self.populations[loser_winner_id] = loser_winner

 

 

population_value = np.linspace(x_bound[0], x_bound[1], 200)

plt.plot(population_value, F(population_value))

 

ga = GA()

#画出图像

for step in range(n_generations):

    population_fitness = ga.calculate_fitness(ga.populations)

    x = ga.DNA_decode(ga.populations[np.argmax(population_fitness)])

    y_max = np.max(population_fitness)

    y_mean = np.mean(population_fitness)



    # globals以字典类型返回当前位置的全部全局变量

if 'sca' in globals():

     sca.remove()

#描最大值点

sca = plt.scatter(x, y_max, s=100, c='red')     #（位置，位置，圈的大小，颜色）

print('the best fitness: %.3f,the best x: %.3f' % (y_max, x))

plt.pause(0.01)    #延续0.01秒后关闭

ga.evolve()

plt.show()

 


 
   #计算适应度
    def calculate_fitness(self, loser_winner):

        DNA_value = self.DNA_decode(loser_winner)

        fitness = F(DNA_value)

        fitness = np.where(fitness < 0.0, 0.0, fitness)

        return fitness

 
#选择
    def selection(self):

        # 从0~DNA.length-1中随机选择2组DNA编号

        loser_winner_id = np.random.choice(np.arange(population_size), size=2, replace=False)

        loser_winner = self.populations[loser_winner_id]

        loser_winner_fitness = self.calculate_fitness(loser_winner)

        return loser_winner[np.argsort(loser_winner_fitness)], loser_winner_id

 

    def crossover(self, loser_winner):

        # 必须指定类型为np.bool，否则True会变为1.0000的小数

        crossover_points = np.empty((DNA_length,)).astype(np.bool)

        for i in range(DNA_length):

            crossover_points[i] = True if random.random() < pc else False

        loser_winner[0, crossover_points] = loser_winner[1, crossover_points]

        return loser_winner

 
#突变
    def mutation(self, loser_winner):

        for i in range(DNA_length):

            loser_winner[0, i] = 1 if loser_winner[0, i] == 0 else 0

        return loser_winner

 
#发展，即进行迭代
    def evolve(self):

        for i in range(int(population_size/2)):

            loser_winner, loser_winner_id = self.selection()

            loser_winner = self.crossover(loser_winner)

            loser_winner = self.mutation(loser_winner)

            self.populations[loser_winner_id] = loser_winner

 

 

population_value = np.linspace(x_bound[0], x_bound[1], 200)

plt.plot(population_value, F(population_value))

 

ga = GA()
#画出图像
for step in range(n_generations):

    population_fitness = ga.calculate_fitness(ga.populations)

    x = ga.DNA_decode(ga.populations[np.argmax(population_fitness)])

    y_max = np.max(population_fitness)

    y_mean = np.mean(population_fitness)



    # globals以字典类型返回当前位置的全部全局变量

if 'sca' in globals():

     sca.remove()
#描最大值点
sca = plt.scatter(x, y_max, s=100, c='red')     #（位置，位置，圈的大小，颜色）

print('the best fitness: %.3f,the best x: %.3f' % (y_max, x))

plt.pause(0.01)    #延续0.01秒后关闭

ga.evolve()

plt.show()

 
