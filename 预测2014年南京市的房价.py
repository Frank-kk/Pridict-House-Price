# -*- coding: utf-8 -*-
"""
Created on Tue Sep 26 14:41:07 2017

@author: dauron
"""

import numpy as np

def geData():
    X=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
    Y=[2.000,2.500,2.900,3.147,4.515,4.903,5.365,5.704,6.853,7.971,8.561,10.000,11.280,12.900]
    points=np.array((X,Y)).T
    return points


def errors(w,b,points):
    totalError=0
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        totalError+=(y-(w*x+b))**2
    return totalError


#计算梯度，偏导数由公式计算得出
def step_gradient(b_current,w_current,points,alpha):
    b_gradient=0
    w_gradient=0
#    N=float(len(points))
    for i in range(0,len(points)):
        x=points[i,0]
        y=points[i,1]
        b_gradient+=(((w_current*x)+b_current)-y)
        w_gradient+=(((w_current*x)+b_current)-y)*x
    new_b=b_current-(alpha*b_gradient)
    new_w=w_current-(alpha*w_gradient)
    return [new_b,new_w]

def gradient_descent_runner(points,starting_b,starting_w,alpha,max_iterations):
    b=starting_b
    w=starting_w
    display=15000
    count=1
    lasterror = 0
    for i in range(max_iterations):
        b,w=step_gradient(b,w,points,alpha)
        if i%display==0:
            print(errors(w,b,points))
            count+=1
            if count%2==0:
                lasterror = errors(w, b, points)
            elif count%2!=0 and np.abs(errors(w,b,points)-lasterror)<0.00000001:
                break
    return [b,w]


def run():
    points=geData()
    alpha=0.00001
    initial_b=10
    initial_w=450
    max_iterations=100000000
    params=gradient_descent_runner(points,initial_b,initial_w,alpha,max_iterations)
    return params

if __name__=='__main__':
    params=run()
    res=params[1]*14+params[0]
    print("预测结果:",res)