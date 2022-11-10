import numpy as np
import random
import math

#子堆场的范围（0,W_size,0,L_size)

def Convert(n):
    # 可以用字典加速，相同大小的n，S也相同
    global tot,x1,x2,y1,y2
    tot = 0
    for r in range(0,r_size):
        for t in range(0,t_size):
            # area = math.ceil(n[r][t]/H)
            area = math.ceil(n[r][t])
            for w1 in range(0,W_size+1):
                for w2 in range(w1+1,W_size+1):
                    width = w2-w1
                    length = math.ceil(area/width)
                    for l1 in range(0,L_size+1):
                        l2 = l1 + length
                        if l2 > L_size:
                            break
                        x1[tot] = w1
                        x2[tot] = w2
                        y1[tot] = l1
                        y2[tot] = l2
                        S[r][t].append(tot)
                        tot += 1
                        # print(tot)
                    if length == 1:
                        break
    # print(tot)
    print("ok")
    x1 = x1[:tot]
    x2 = x2[:tot]
    y1 = y1[:tot]
    y2 = y2[:tot]
    return S

def Init_Generate():
    # print(num)
    temp = sorted(num.items(), key=lambda d: -d[1], reverse=False)
    # print(temp)
    p = []
    for item in temp:
        p.append(item[0])
    return p

def overlap(s,status):
    # print(x1[s],x2[s],y1[s],y2[s])
    for w in range(x1[s],x2[s]):
        for l in range(y1[s],y2[s]):
            if(status[w][l] == 1):
                return True
    return False

def cover(s_ , s):
    if(x1[s_]>x1[s]):
        return False
    elif(y1[s_]>y1[s]):
        return False
    elif(x2[s_]<x2[s]):
        return False
    elif(y2[s_]<y2[s]):
        return False
    else:
        return True

def solverequest(r,w,pre_L,pre_L_max,status):
    # 第i个request固定w求最小的l
    ss = [0 for j in range(t[r]-a[r]+1)]
    nxt = [[0 for i in range(tot)]for j in range(t[r]-a[r]+1)]
    h = [[inf for i in range(tot)]for j in range(t[r]-a[r]+1)]
    L = inf
    L_max = 0
    for i in range(L_size):
        for j in range(w):
            L_max = n[r][t[r]] + i
            break
        if L_max:
            break
    L_max = max(pre_L_max,L_max)
    # print(w,L_max)
    L_max = min(L_size,L_max)
    while L == inf and L_max<=L_size:
        for k in range(t[r]-a[r],-1,-1):
            for s in S[r][k+a[r]]:
                # print(n[r][k+a[r]])
                # print(x1[s],x2[s],y1[s],y2[s])
                # print()
                if x2[s] > w or y2[s] > L_max:
                    continue
                elif overlap(s,status[k+a[r]]) == True : #初始值为inf，如果k+1没有合适的s_，h[k][s]==inf
                        # print(1)
                        continue
                elif k == t[r]-a[r]:
                    # print(x1[s],x2[s],y1[s],y2[s])
                    # print(overlap(s, status))
                    h[k][s] = max(pre_L,y2[s])
                    # print(h[k][s])
                else:
                    for s_ in S[r][k+a[r]+1]:
                        # print(x1[s],x2[s],y1[s],y2[s])
                        # print(x1[s_],x2[s_],y1[s_],y2[s_])
                        # print(cover(s_,s))
                        if h[k+1][s_] == inf:
                            continue
                        if cover(s_,s) == False:
                            continue
                        # print(h[k][s],h[k+1][s_])
                        if h[k][s]>h[k+1][s_]:
                            h[k][s] = h[k+1][s_]
                            # print(h[k][s])
                            nxt[k][s] = s_
                        # print(h[k][s])
                        # return
                if(k == 0):
                    # print(h[k][s])
                    if(L > h[k][s]):
                        L = h[k][s]
                        ss[k] = s
        # print(h[k][s])
        L_max = L_max + 1
    for k in range(0,t[r]-a[r]):
        temp = ss[k]
        ss[k+1] = nxt[k][temp]

    # ss是第i个request每个时间段选择的空间编号
    return L_max,L,ss

def updateyardstatus(r,s,status):
    # print(s)
    # print(a[r],t[r])
    for time in range(a[r],t[r]+1):
        i = s[time-a[r]]
        for w in range(x1[i],x2[i]):
            for l in range(y1[i],y2[i]):
                if(status[time][w][l] == 1):
                    print(time,w,l)
                status[time][w][l] = 1
    return status
    

def DSAP(p):
    # L,W, size is (1,N)
    # Omega size is (1,N) = inf
    pre_L = 0
    pre_L_max = 0
    pre_W = 1
    Omega = inf
    # s = np.zeros((W_size,L_size))
    s = [[]for i in range(r_size)]
    status = [[[0 for i in range(L_size)]for j in range(W_size)] for t in range(t_size)]
    for i in range(r_size):
        L = 0
        L_max = 0
        W = 1
        Omega = inf
        r = p[i]
        for w in range(W_size,pre_W-1,-1):
            # temp_s = {s1,……,sTr}
            # s1 = {x-,x+,y-,y+}
            temp_L_max,temp_L,temp_s = solverequest(r,w,pre_L,pre_L_max,status)
            # print(temp_L)
            temp_Omega = temp_L * w
            if(temp_Omega < Omega):
                Omega = temp_Omega
                # print(temp_s)
                L = temp_L
                L_max = temp_L_max
                W = w
                s[i] = temp_s
            # break
        pre_L = L
        pre_W = W
        pre_L_max = L_max
        status = updateyardstatus(r,s[i],status)
        # print(s[i])
    print(s,Omega)
    return s,Omega

def Generate(p):
    temp = random.sample(range(0,len(p)), 2)
    p[temp[0]], p[temp[1]] = p[temp[1]], p[temp[0]]
    return p

def SA_DSAP(n):
    S = Convert(n)
    # print(tot,len(x1))
    p = Init_Generate()
    s , Omega = DSAP(p)
    # return s, p, Omega
    p_best = p
    s_best = s
    Omega_best = Omega
    temperature = alpha * Omega
    temperature_schedule = []
    while temperature >= 0.001:
        temperature_schedule.append(temperature)
        temperature = beta * temperature
    for temperature in temperature_schedule:
        K = K_max
        I = 0
        while K>0:
            # print(p)
            p_ = Generate(p)
            # print(p_)
            s_ , Omega_ = DSAP(p_)
            if Omega_ < Omega_best:
                I = 0
            else:
                I = I + 1
            if I >= I_max:
                break
            if Omega_ < Omega:
                p = p_
                Omega = Omega_
                s = s_
                if Omega < Omega_best:
                    p_best = p
                    Omega_best = Omega
                    s_best = s
            elif random.uniform(0,1) < np.exp((Omega - Omega_)/temperature) :
                p = p_
                Omega = Omega_
                s = s_
            # if(Omega_best == Lower_Bound):
            #     return s_best , p_best , Omega_best
            K = K-1
    return s_best , p_best , Omega_best

inf = float("inf")
f = open  ("./3dyapt-master/test",  "r")
data_list =  f.readlines()
f.close()

r_size = eval(data_list[0].strip())
data_list = data_list[1:]
t_size = 168 + 1
alpha = 0.382
beta = 0.618
K_max = 500
I_max = 400
# 该算法一次分配能处理的规模受数组大小限制。
# t_size最大不能超过数组能够装下的大小,可以分批处理不同时间长度的数据

t = [0 for i in range(r_size)]
a = [0 for i in range(r_size)]
n = [[0 for i in range(t_size)] for j in range(r_size)]

num = {}
for i in range(r_size):
    a[i] , t[i] = data_list[2*i].strip().split(' ')
    a[i] = eval(a[i])
    t[i] = eval(t[i])
    temp = data_list[2*i+1].strip().split(' ')
    for time in range(a[i],t[i]+1):
        n[i][time] = eval(temp[time-a[i]])
    num[i] = n[i][t[i]]

Lower_Bound = 0
for time in range(t_size):
    temp = 0
    for r in range(r_size):
        temp = temp + n[r][time]
    Lower_Bound = max(Lower_Bound,temp)
print(Lower_Bound)

H = 5
W_size = 6
L_size = 15
S = [[[]for i in range(t_size)]for j in range(r_size)]
x1 = [0 for i in range(r_size*t_size*W_size*(W_size-1)*L_size*(L_size)//4)]
x2 = [0 for i in range(r_size*t_size*W_size*(W_size-1)*L_size*(L_size)//4)]
y1 = [0 for i in range(r_size*t_size*W_size*(W_size-1)*L_size*(L_size)//4)]
y2 = [0 for i in range(r_size*t_size*W_size*(W_size-1)*L_size*(L_size)//4)]

s, p, Omega = SA_DSAP(n)

status = [[[0 for i in range(L_size)]for j in range(W_size)] for t in range(t_size)]

for num in range(len(p)):
    r = p[num]
    item = s[num]
    for i in item:
        for time in range(a[r],t[r]+1):
            for w in range(x1[i],x2[i]):
                for l in range(y1[i],y2[i]):
                    status[time][w][l]=1

for time in range(31):
    print(status[time])



# def solverequest(i,p,R,w,L,s_0_i):
#     # 固定w求最小的l
#     # L is int
#     # p is request list
#     # 
#     r = p[i]
#     T[r] = t[r] - a[r] + 1
#     for k in range(T[r],0,-1):
#         for s in S[r][k+a[r]-1]:
#             # s = [x-,x+,y-,y+]
#             if k == 0:
#                 h[k][s] = L
#             elif overlap() or decrease():
#                 h[k][s] = M
#             elif k == T[r]:
#                 h[k][s] = max(L,s[4])
#             else:
#                 h[k][s] = min_h[k+1]
#     L[i][w] = min(h[1][s])
#     s[i][w] = argmin(h[1][s])
#     return L[i][w],s[i][w]


# def DSAP(p,R,W,L):
#     # s[i]是决定第i个request后的每个unit被占领的情况
#     L[0] = 0
#     W[0] = 1
#     s[0] = np.zeros((W,L)) # W * L的0，1矩阵
#     for i in range(1,N+1):
#         for w in range(W,W[i-1]-1,-1):   #枚举w的大小
#             L[i][w] , s[i][w] = solverequest(i,p,R,w,L[i-1],s[0:i-1]) #固定w的大小后，算出最优长度。
#             Omega[i][w] = L[i][w] * w
#         Omega[i] = min(Omega[i][w])
#         W[i] = argmin(Omega[i][w])
#         L[i] = L[i][W[i]]
#         s[i] = s[i][W[i]]
#         updateyardstatus
#     s = s[1:N] , Omega = Omega[N]
#     return s,Omega