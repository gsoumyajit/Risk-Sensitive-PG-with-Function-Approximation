import numpy as np
from env2 import CustomEnv
from scipy.special import softmax
from numpy.random import randint,choice
from collections import deque
from scipy.stats.mstats import gmean
import os
from tqdm import tqdm
import time

nS=100
nA=9
ff=10
d=nS//ff
d1=d*nA
value=np.ones(d)
value1=np.zeros((d1,d))
u=np.zeros((d1,d))
theta=np.zeros(d1)
logrd="data/acff1/"
if not os.path.exists(logrd):
          os.makedirs(logrd)
run_num = len(next(os.walk(logrd))[2])
fr=open(logrd+str(run_num)+".csv","w")

A=0
Bi=np.eye(d)
env=CustomEnv()
state=env.reset()
N=100000000
def feat(state):
    res=np.zeros(d)
    res[state//ff]=1
    return res
def phi(state,action):
    res=np.zeros(d*nA)
    res[d*action:d*(action+1)]=feat(state)
    return res
def correct(a):
    return max(a,0.0001)

start_time=time.time()
fr.write("timestep\treward\treward1\n")
returns=deque(maxlen=1000000)
state=state0=env.reset()
k=0.001
K=100000000
for n in range(N):
    a=0.1/(n//K+1)**0.55
    b=0.03/(n//K+1)**0.8
    c=0.01/(n//K+1)**1
    probs=softmax([np.dot(theta,phi(state,k)) for k in range(nA)])
    action=choice(nA,p=probs/np.sum(probs))

    next_state,reward,_,_=env.step(action)
    reward=k*reward
    A+=np.exp(reward)*np.outer(feat(state),feat(next_state))
    Bi-=np.outer(np.dot(Bi,feat(next_state)),np.dot(feat(next_state),Bi))/(1+np.dot(np.dot(feat(next_state),Bi),feat(next_state)))
    value+=a*(np.dot(Bi,np.dot(A,value))/correct(np.dot(feat(state0),value))-value) 
    psi=phi(state,action)-sum([phi(state,k)*probs[k] for k in range(nA)])
    factor=np.exp(reward)*np.dot(value,feat(next_state))/correct(np.dot(value,feat(state))*np.dot(value,feat(state0)))
    
    delta=(factor-1)*psi+factor*np.dot(value1,feat(next_state))-np.dot(value1,feat(state))-np.dot(value1,feat(state0))
    u+=b*np.outer(delta-np.dot(u,feat(state)),feat(state))
    value1+=b*np.outer(np.dot(u,feat(state)),feat(state)-factor*feat(next_state)+feat(state0))
    
    theta-=c*np.dot(value1,feat(state0))
    returns.append(reward/k)
    if n%1000000==0:
        #e,v=np.linalg.eig(np.matmul(A,Bi))
        print(probs)
        #print(v[:,1]*e[1]/v[0,1],value)
        print(n//1000000,state,np.mean(returns),np.std(returns),np.linalg.norm(theta),np.linalg.norm(value),np.linalg.norm(value1))
        fr.write(str(n)+'\t'+str(np.mean(returns))+'\t'+str(np.std(returns))+'\n')
        fr.flush()

    state=next_state
end_time=time.time()
print("Elapsed time:",end_time-start_time)
fr.close()
