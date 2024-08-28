# -*- coding: utf-8 -*-
import argparse
import random
import socket
import time
from collections import namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def send_message(sock, message):
    message_with_delimiter = message
    sock.sendall(message_with_delimiter.encode())

def recv_message(sock):
    buffer = bytearray()
    while True:
        data = sock.recv(1)
        if not data or data == b'\0':
            break
        buffer.extend(data)
    return buffer.decode()

# 定义Actor网络
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        # 定义第一层全连接层，输入维度为state_dim，输出维度为hidden_dim
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        # 定义第二层全连接层，输入维度为hidden_dim，输出维度为state_dim
        self.fc2 = nn.Linear(hidden_dim, state_dim)
        # 定义第三层全连接层，输入维度为state_dim，输出维度为action_dim
        self.fc3 = nn.Linear(state_dim, action_dim)

    def forward(self, x):
        # 使用leaky_relu激活函数，负斜率为0.01
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        # 使用leaky_relu激活函数，负斜率为0.01
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        # 使用tanh激活函数
        x = torch.tanh(self.fc3(x))
        # 对输出进行clamp操作，限制在-1到1之间
        # x=torch.clamp(x,min=-1,max=1)
        # 打印输出
        # print('x:', x)
        return x


# 定义Critic网络
class Critic(nn.Module):
    # 定义Critic类，继承自nn.Module
    def __init__(self, state_dim, action_dim, hidden_dim):
        # 初始化函数，传入状态维度、动作维度和隐藏层维度
        super(Critic, self).__init__()
        # 调用父类的初始化函数
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        # 定义第一个全连接层，输入维度为状态维度和动作维度之和，输出维度为隐藏层维度
        self.fc2 = nn.Linear(hidden_dim, state_dim + action_dim)
        # 定义第二个全连接层，输入维度为隐藏层维度，输出维度为状态维度和动作维度之和
        self.fc3 = nn.Linear(state_dim + action_dim, 1)
        # 定义第三个全连接层，输入维度为状态维度和动作维度之和，输出维度为1

    def forward(self, state, action):
        # 定义前向传播函数，传入状态和动作
        x = torch.cat([state, action], dim=1)
        # 将状态和动作按列拼接
        x = F.leaky_relu(self.fc1(x), negative_slope=0.01)
        # 将拼接后的结果通过第一个全连接层，并使用leaky_relu激活函数，负斜率为0.01
        x = F.leaky_relu(self.fc2(x), negative_slope=0.01)
        # 将结果通过第二个全连接层，并使用leaky_relu激活函数，负斜率为0.01
        x = self.fc3(x)
        # 将结果通过第三个全连接层
        return x
        # 返回结果



class TD3Agent:
    def __init__(self, state_dim, action_dim, hidden_dim, gamma, tau, lr_actor, lr_critic, policy_noise, noise_clip,
                 policy_delay):
        # 初始化Actor网络
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        # 初始化目标Actor网络
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        # 初始化Critic1网络
        self.critic1 = Critic(state_dim, action_dim, hidden_dim)
        # 初始化目标Critic1网络
        self.target_critic1 = Critic(state_dim, action_dim, hidden_dim)
        # 初始化Critic2网络
        self.critic2 = Critic(state_dim, action_dim, hidden_dim)
        # 初始化目标Critic2网络
        self.target_critic2 = Critic(state_dim, action_dim, hidden_dim)
        # 初始化经验回放内存
        self.memory = ReplayMemory()  
        # 初始化折扣因子
        self.gamma = gamma  
        # 初始化软更新参数
        self.tau = tau  
        # 初始化策略噪声标准差
        self.policy_noise = policy_noise  
        # 初始化噪声裁剪范围
        self.noise_clip = noise_clip  
        # 初始化策略延迟更新步数
        self.policy_delay = policy_delay  
        # 初始化Actor网络的优化器
        self.optimizer_actor = optim.Adam(self.actor.parameters(), lr=lr_actor, weight_decay=0.001)
        # 初始化Critic1网络的优化器
        self.optimizer_critic1 = optim.Adam(self.critic1.parameters(), lr=lr_critic, weight_decay=0.001)
        # 初始化Critic2网络的优化器
        self.optimizer_critic2 = optim.Adam(self.critic2.parameters(), lr=lr_critic, weight_decay=0.001)
        # 初始化迭代次数
        self.total_it = 0  
        # 初始化探索率
        self.ep = 0  
        # 初始化探索率的衰减率
        self.epsa = 0.995  
        # 初始化探索率的最小值
        self.lim = 0.8  


    def save_models(self):
    # 定义模型和文件名的字典
        models_to_save = {
            "actor": "actor3.pth",
            "target_actor": "target_actor3.pth",
            "critic1": "critic13.pth",
            "target_critic1": "target_critic13.pth",
            "critic2": "critic23.pth",
            "target_critic2": "target_critic23.pth"
        }

        # 保存模型状态字典
        for model_name, filename in models_to_save.items():
            model_state_dict = getattr(agent, model_name).state_dict()
            torch.save(model_state_dict, filename)
        print('sucessful save')

    def select_action(self, state):
        return self.exploit(state) 


    def exploit(self, state):
        # action三个量为速度，偏移位置，角速度
        # state35个量，前32个为位置量，33号为回合数，34号为得分数，35号为对方场上存在的球数
        # action为归一化量，速度在-1-1之间，-1到-0.5代表2.4到2.8为护住球，-0.5到0.5代表2.8到3.2为得分球，0.5到1代表3.2到6为撞击球
        # action偏移位置-1到1代表-2到2，角速度-1到1代表-3到3
        # state中位置为归一化量，范围在0-1，要正常使用x请乘以4.2996，y乘以10.4154。回合数也为归一化量，范围0-1，正常使用乘以16。
        # state中得分也为归一化量范围0-1，正常使用乘以8。对方球数为归一化量，范围0-1，正常使用乘以8
        default_action = np.zeros(3)  # 假设动作维度为3
        #print('episode:{}'.format(state[32]))
        score = state[33]
        z = [2.375, 4.88]
        z1=2.3993
        p1 = 0
        p2 = 0
        houshou = (state[32] * 16) % 2 == 0
        # xy1为我方最好一球的坐标
        # xy2为对方最好一球的坐标
        xy1 = getxy(not houshou, state[0:32])  # 先手计算撞后手的位置，一下同理
        xy2 = getxy(houshou, state[0:32])
        # 如果先手有可撞的位置且分数小于0，则计算先手的概率
        if len(xy1) > 0 and score < 0:
            p1 = getp(xy1, score, state[32],houshou)
            #print('p1',p1)
        # 如果后手有可撞的位置且分数大于0，则计算后手的概率
        if len(xy2) > 0 and score > 0:
            p2 = getp(xy2, score, state[32],houshou)
            #print('p2',p2)
            '''
            # 将状态转换为PyTorch张量
            state = torch.FloatTensor(state).unsqueeze(0)
            # 不计算梯度
            with torch.no_grad():
                # 使用actor网络计算动作
                action = self.actor(state).squeeze(0).numpy()
            # 添加噪声
            noise = np.random.normal(0, self.policy_noise, size=action.shape)
            action += noise
            # print('action:', action)
            print('对方圈内有球+局面优势，网络决策')
            return np.clip(action, -1, 1)  
            '''
        re=[]
        re = self.strategy_first(state)#人工策略1：先手固定位置开球
        if len(re) != 0:
            return re
        re = self.strategy_second(state)#人工策略2：后手开球护住
        if len(re) != 0:
            return re
        re = self.strategy_tenth(state)#人工策略10：简单先手第二颗球造球
        if len(re) != 0:
            return re
        re = self.strategy_eleventh(state)# 人工策略11：简单后手第二颗球阻拦
        if len(re) != 0:
            return re
        re = self.strategy_ninth(state)#人工策略9：侧边撞人
        if len(re) != 0:
            return re
        re = self.strategy_fifteenth(state)#人工策略15：滑翔预测
        if len(re) != 0:
            return re  
        re = self.strategy_twelfth(state)#人工策略12：递推预测
        if len(re) != 0:
            return re
        re = self.strategy_third(state)#人工策略3：把对手撞飞。比分低的情况下启动，越低和回合数过去越多，启动概率越高
        if len(re) != 0:
            return re
        re = self.strategy_thirteenth(state)#人工策略13：见缝插针加递推预测
        if len(re) != 0:
            return re
        re = self.strategy_fifth(state)#人工策略5：后手方见缝插针，在我方劣势时启动
        if len(re) != 0:
            return re
        re = self.strategy_seventh(state)#人工策略7，反见缝插针
        if len(re) != 0:
            return re
        re = self.strategy_fourth(state)#人工策略4：护住得分球。比分高的情下启动，得分越高启动概率越高
        if len(re) != 0:
            return re
        re = self.strategy_forteenth(state)#人工策略14：优势局或均势局中路造球
        if len(re) != 0:
            return re
        re = self.strategy_eighth(state)#人工策略8，反侧边隔山打牛球
        if len(re) != 0:
            return re
        re = self.strategy_default(state)#上述情况以外（网络决策）
        if len(re) != 0:
            return re
        
    def strategy_twelfth(self, state):
        #人工策略12：递推预测
        # 把自己在中心下方的球打入
        z = [2.375, 4.88]
        s = []
        score = state[33]
        if  ((state[32]*16)!=1 and (state[32]*16)!=16 and score>=0):
            houshou = (state[32] * 16) % 2 == 0
            # 自己第一球在状态中的起始索引
            start = 2 if houshou else 0
            # 对方最好的一球坐标
            xy = getxy(houshou, state[0:32])
            s = []
            flag=False
            # 对方有球进入环内
            if len(xy)>0:
                dis1=dist(xy,z)
            else:
                dis1=1.975
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
                s.append(state[i + 1] * 10.4154)
            d=[]
            v=[]
            if  not flag:
                for i in range(start, 32, 4):
                    # 我方的球在中心下面且左右偏移不超过1.875能进环
                    if  s[i + 1] >= 5.46 and z[0] - 1.975 <= s[i] <= z[0] + 1.975:
                        # 我方球和中心的x方向距离
                        dis=dist([s[i],z[1]],z)
                        # 如果离中心太远就不送进去
                        if dis>=dis1:
                            continue
                        flag = True
                        # 对方的球在中心上面且距离我方球很近就不送进去
                        for j in range(0, 32, 2):
                            if j == i:
                                continue
                            if s[i] - 0.29 <= s[j] <= s[i] + 0.29 and s[j + 1] >z[1]-0.33:
                                flag = False
                                break
                        # 若对方没有球离我方球很近的话
                        if flag:
                            v.append(getv(s[i+1]))
                            if s[i] > z[0]:
                                d.append((s[i] - 2.375 + 0.04) / 2)
                            else:
                                d.append((s[i] - 2.375 - 0.02) / 2)
                # 若有球位于中心上方且我方没有球离对方很近
                # 此时把自己的球送进大环
                if len(d) > 0:
                    d1 = d[0]
                    v1 = v[0]
                    for i in range(0, len(d)):
                        if abs(d[i]) < abs(d1):
                            d1 = d[i]
                            v1 = v[i]
                    print('把自己球送入圈内')
                    return np.array([v1, d1, 0])#人工策略12结束
        return []
                
    def strategy_fifteenth(self, state):
        #人工策略15,滑翔预测
        # 疑似想用中心两边区域不管哪边的球挡一下自己的球，让打出去的这球弹到得分区
        z = [2.375, 4.88]
        s = []
        if (state[32]*16)!=1 and (state[32]*16)!=16:
            xy = z
            houshou = (state[32]*16)%2 == 0
            # 对方壶的坐标起始索引
            start = 0 if houshou else 2
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
                s.append(state[i + 1] * 10.4154)
            dis=dist(xy,z)
            if dis<=1.22:
                '''# 遍历所有球作为打击的备选
                for i in range(0,32,2):'''
                # 遍历对方的球作为打击的备选
                for i in range(start,32,4):
                    #在中心下方的两侧
                    #if (s[i+1]>xy[1]  and s[i+1]<xy[1]+1.875)  and abs(s[i]-xy[0])>0.6 and abs(s[i]-z[0])<1.86 :
                    if (s[i+1]>xy[1]  and s[i+1]<xy[1]+4)  and abs(s[i]-xy[0])>0.6:
                        s1=[s[i],s[i+1]]
                        # 离中心点所在横线太远或者太近
                        if abs((s1[1]-xy[1])/(s1[0]-xy[0]))<0.1 or abs((s1[1]-xy[1])/(s1[0]-xy[0]))>2:
                            continue
                        flag=True
                        for j in range(0,32,2):
                            if j==i:
                                continue        # 跳过自己
                            dis=dist([s[j],s[j+1]],s1)
                            # 两球接近
                            if dis <=0.4:
                                flag=False
                                break
                            # 靠近备选球在我方行进路径上
                            #if  s1[0]-0.455<=s[j]<=s1[0]+0.455 and s[j+1]>s1[1]-0.33:
                            if  s1[0]>z[0] and s1[0] - 0.29<=s[j]<=s1[0] + 0.145 and s[j+1]>s1[1]:
                                flag=False
                                break
                            if  s1[0]<z[0] and s1[0] + 0.29>=s[j]>=s1[0] - 0.145 and s[j+1]>s1[1]:
                                flag=False
                                break
                            # 两球连线经过中心附近，可能碰到自己球
                            if getc(s1,[s[j],s[j+1]],xy):
                                flag=False
                                break
                        b = getb1(s1, xy)
                        if flag and abs(b)<=0.32:
                            v=getv2(s1)
                            print('不使用网络决策，滑翔')
                            #return np.array([v,(s1[0]-z[0]+b)/2,0])#人工策略15结束
                            if s1[0]>xy[0]:
                                return np.array([v,(s1[0]-z[0]+b+0.01)/2,0])#人工策略15结束
                            else:
                                return np.array([v,(s1[0]-z[0]+b-0.01)/2,0])#人工策略15结束
        return []
               
    def strategy_thirteenth(self, state):                    
        #人工策略13：见缝插针加递推预测
        # 第12，13回合，对方得分时
        # 在中心附近有两球之间有一定间隔时，从间隔里蹭过去得分
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if (((state[32]*16)==13 or (state[32]*16)==14) and score>1): 
            # 对方最好的一球坐标
            houshou = (state[32]*16)%2==0
            xy = getxy(houshou, state[0:32])
            if len(xy) > 0:
                dis = dist(xy, z)
            else:
                dis = 0.71
            if dis >=0.71:
                dis=0.71
            s = [z[0] - dis]
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
            s.append(z[0] + dis)
            s = sorted(s)
            for i in range(0, len(s) - 1):
                if z[0] - dis <= s[i] <= z[0] + dis and z[0] - dis <= s[i + 1] <= z[0] + dis:
                    if s[i + 1] - s[i] >= 0.66:
                        print('不使用网络决策，见缝插针')
                        #return np.array([-0.58, (s[i] + 0.33 - 2.375) / 2, 0]) #人工策略13结束
                        return np.array([0, (s[i] + 0.33 - 2.375) / 2, 0]) #人工策略13结束
        return []
                
    def strategy_ninth(self, state):     
        #人工策略9,侧边撞人
        # 似乎想要通过撞击s1（基本在对方最好球右侧一片区域），力大砖飞侧边撞开对方好球
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if score<0 and ((state[32]*16)!=1 and (state[32]*16)!=16 ):
            # 对方最好的一球坐标，start从我开始
            houshou = (state[32]*16)%2==0
            xy = getxy(houshou, state[0:32])
            start = 2 if houshou else 0
            s = []
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
                s.append(state[i + 1] * 10.4154)
            flag=False
            # 存在一球基本在对方最好一球下方
            for i in range(0,32,2):
                if xy[0]-0.29<=s[i]<=xy[0]+0.29 and s[i+1]>xy[1]:
                    flag=True
                    break
            dis=dist(xy,z)
            if dis<=1.22 and flag:
                for i in range(0,32,2):
                    # 筛选在对方最好球后方2.5内，不直接在正后方，且x方向偏离中心不超过1.8的球
                    #if (s[i+1]>xy[1]  and s[i+1]<xy[1]+2.5)  and s[i]!=xy[0] and abs(s[i]-z[0])<1.86:
                    if (s[i+1]>xy[1]  and s[i+1]<xy[1]+3)  and s[i]!=xy[0] and abs(s[i]-z[0])<1.975:
                        s1=[s[i],s[i+1]]
                        #if abs((s1[1]-xy[1])/(s1[0]-xy[0]))<0.9 or abs((s1[1]-xy[1])/(s1[0]-xy[0]))>2:
                        if abs((s1[1]-xy[1])/(s1[0]-xy[0]))<0.1 or abs((s1[1]-xy[1])/(s1[0]-xy[0]))>2:
                            continue
                        flag=True
                        for j in range(0,32,2):
                            if j==i:
                                continue
                            dis=dist([s[j],s[j+1]],s1)
                            if dis <=0.4:
                                flag=False
                                break
                            #if  s1[0]-0.435<=s[j]<=s1[0]+0.435 and s[j+1]>s1[1]-0.33:
                            '''if  s1[0]-0.425<=s[j]<=s1[0]+0.425 and s[j+1]>s1[1]-0.33:
                                flag=False
                                break'''
                            if  s1[0]>z[0] and s1[0] - 0.145<=s[j]<=s1[0] + 0.29 and s[j+1]>s1[1]:
                                flag=False
                                break
                            if  s1[0]<z[0] and s1[0] + 0.145>=s[j]>=s1[0] - 0.29 and s[j+1]>s1[1]:
                                flag=False
                                break
                            if getc(s1,[s[j],s[j+1]],xy):
                                flag=False
                                break
                        if i % 4 == start:  # 我方的球
                            if flag:
                                print('我方侧边')
                                b = getb(s1, xy)   
                                return np.array([0.97, (s1[0] - z[0] + b) / 2, 0])                       
                                '''if s1[0] < xy[0]:
                                    #return np.array([0.97, (s1[0] - z[0] - b - 0.05) / 2, 0])                                   
                                else:
                                    #return np.array([random.uniform(0.95, 1), (s1[0] - z[0] + 0.117 + b) / 2, 0])'''
                        else:  # 敌方的球
                            if flag:
                                print('对方侧边')
                                b = getb(s1, xy)
                                return np.array([0.97, (s1[0] - z[0] + b) / 2, 0])  
                                '''if s1[0] < xy[0]:
                                    #eturn np.array([random.uniform(0.95, 1), (s1[0] - z[0] - b - 0.05) / 2, 0])
                                    #return np.array([0.97, (s1[0] - z[0] - b - 0.05) / 2, 0])
                                else:
                                    #return np.array([0.97, (s1[0] - z[0] + 0.117 + b) / 2, 0])#人工策略9结束'''
        return []
                            
    def strategy_fifth(slef, state):
        # 人工策略5：后手方见缝插针
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if state[32] * 16 == 16 : 
            # 对方最好球 
            xy = getxy(True, state[0:32])
            if len(xy) > 0:
                dis = dist(xy, z)
            else:
                dis = 1.875
            s = [z[0] - dis]
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
            s.append(z[0] + dis)
            s = sorted(s)
            # 判断是否有大缝
            for i in range(0, len(s) - 1):
                if z[0] - dis <= s[i] <= z[0] + dis and z[0] - dis <= s[i + 1] <= z[0] + dis:
                    if s[i + 1] - s[i] >= 0.66:
                        print('大缝，后手见缝插针')
                        return np.array([0, (s[i] + 0.33 - 2.375) / 2, 0])
            if score < 0:
                s1 = []
                for i in range(0, 32, 2):
                    s1.append(state[i] * 4.2996)
                    s1.append(state[i + 1] * 10.4154)
                count=int(score/(-0.125))
                for i in range(0,count):
                    flag = False
                    x=xy[0]
                    count1=0
                    for j in range(0, 32, 2):
                        # 统计靠近对方最好球的数量
                        if xy[0] - 0.29 <= s1[j] <= xy[0] + 0.29 and s1[j + 1] > xy[1]:
                            count1+=1
                            if count1>=2:
                                flag = True
                                break
                            x=s1[j]
                    # 对方最好球周围只有一颗球，攻击
                    if not flag:
                            print('对方最好球周围只有一颗球，攻击')
                            return np.array([0.97, (x - z[0] +0.03) / 2, 0])
                    dis=dist(xy,z)
                    dis1=1.875
                    for j in range(0,32,4):
                        dis2=dist([s1[j],s1[j+1]],z)
                        if dis2>dis and dis2<dis1:
                            dis1=dis2
                            xy=[s1[j],s1[j+1]]
                dis = []
                # 对方所有球
                for i in range(0, 32, 4):
                    dis1 = dist([s1[i], s1[i + 1]], z)
                    if dis1 <= 1.875:
                        dis.append(dis1)
                    else:
                        dis.append(1.875)
                dis = sorted(dis)
                for i in range(1, 8):
                    s = s1[0:32:2]
                    s.append(z[0] - dis[i])
                    s.append(z[0] + dis[i])
                    s = sorted(s)
                    for j in range(0, len(s) - 1):
                        if z[0] - dis[i] <= s[j] <= z[0] + dis[i] and z[0] - dis[i] <= s[j + 1] <= z[0] + dis[i]:
                            if s[j + 1] - s[j] >= 0.66:
                                print('后手见缝插针')
                                return np.array([0, (s[j] - 2.375 + 0.33) / 2, 0])  # 人工策略5：结束
        return []
    
    def strategy_seventh(self, state):                        
        #人工策略7，反见缝插针
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if (state[32]*16)==15 and score>0 :
            xy=getxy(True,state[0:32])
            if len(xy)>0:
                flag=False
                s=[]
                for i in range(0,32,2):
                    s.append(state[i]* 4.2996)
                    s.append(state[i+1]* 10.4154)
                for i in range(0,32,2):
                    if  xy[0]-0.29<=s[i]<=xy[0]+0.29 and s[i+1]>=xy[1]+2.1:#最好与下方护球同样的判断准则
                        flag=True
                        break
                if flag:
                    dis=dist(xy,z)
                    s1=s[0:32:2]
                    s1.append(z[0]-dis)
                    s1.append(z[0]+dis)
                    s1=sorted(s1)
                    d=[]
                    for i in range(0,len(s1)-1):
                        if z[0]-dis<=s1[i]<=z[0]+dis and z[0]-dis<=s1[i+1]<=z[0]+dis:
                            if s1[i+1]-s1[i]>=0.66:
                                dd=s1[i+1]-s1[i]
                                d.append((s1[i]+dd/2-z[0])/2)
                    if len(d)>0:
                        mi=d[0]
                        for i in range(0,len(d)):
                            if abs(d[i])<abs(mi):
                                mi=d[i]
                        print('反见缝插针')                        
                        return np.array([-0.7,mi,0]) #人工策略7：结束
        return []
                
    def strategy_first(self, state):
        # 人工策略1：先手固定位置开球
        # 固定一个护球位置
        #if state[32]*16 == 1 or state[32] == 1: 
        if state[32]*16 == 1: 
            action = np.array([-0.56654, 0, 0])
            print('先手固定位置开球')
            return np.clip(action, -1, 1)# 人工策略1：到此结束，可以选择注释，或者修改
        return []

    def strategy_second(self, state):
        # 人工策略2：后手开球护住
        score = state[33]
        z = [2.375, 4.88]
        if state[32] * 16 == 2 and score == 0:
            # 对方第一球  
            xy=[state[0]*4.2996,state[1]*10.4154]
            if z[1]<=xy[1]<=z[1]+2.5 and abs(xy[0]-z[0])<=1.875:
                action = np.array([getv1(xy[1])-0.1, (xy[0]-z[0]-0.05)/2, 0])
            else:
                action = np.array([-0.56654, 0, 0])
            print('后手开球')
            return np.clip(action, -1, 1)# 人工策略2：到此结束
        return []
    
    def strategy_tenth(self, state):
        #人工策略10：简单先手第二颗球造球
        score = state[33]
        z = [2.375, 4.88]
        if state[32]*16==3 and score==0:
            # 后手第一颗的横坐标
            x=state[2]*4.2996
            if x<=z[0]:
                #action=np.array([-0.5664,0.7,0])
                action=np.array([-0.5664,0.75,0])
            else:
                #action=np.array([-0.5664,-0.7,0])
                action=np.array([-0.5664,-0.75,0])
            print('先手第二颗')
            return action #人工策略10结束
        return []

    def strategy_eleventh(self, state):
        #人工策略11：简单后手第二颗球阻拦
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if state[32]*16==4 and score==0: 
            xy = [state[4] * 4.2996, state[5] * 10.4154]
            if z[1]<=xy[1]<=z[1]+2.5 and abs(xy[0]-z[0])<=1.875:
                flag=True
                for i in range(0,6,2):
                    s=[state[i] * 4.2996, state[i+1] * 10.4154]
                    if abs(s[0]-xy[0])<=0.29 and s[1]>xy[1]:
                        flag=False
                        break
                if flag:
                    print('后手第二颗')
                    return np.array([getv1(xy[1])-0.1,(xy[0]-z[0]-0.05)/2,0])#人工策略11结束
        return []
            
    def strategy_third(self, state):
        # 人工策略3：把对手撞飞。比分低的情况下启动，越低和回合数过去越多，启动概率越高
        score = state[33]
        z = [2.375, 4.88]
        s = []
        count = 0
        houshou = (state[32] * 16) % 2 == 0
        # xy1为我方最好一球的坐标
        xy1 = getxy(not houshou, state[0:32])  # 先手计算撞后手的位置，一下同理
        # 如果先手有可撞的位置且分数小于0，则计算先手的概率
        if len(xy1) > 0 and score < 0:
            p1 = getp(xy1, score, state[32], houshou)
        if score < 0:  
            #对方最好壶
            houshou = (state[32]*16)%2 == 0
            xy = getxy(houshou, state[0:32])
            flag=False
            y=xy[1]
            x=xy[0]
            for i in range(0, 32, 2):
                s=[state[i] * 4.2996,state[i + 1] * 10.4154]
                # 有一球在对方最好壶（或者中介球）下方附近
                # if x - 0.39 <= s[0] <= x + 0.39 and s[1] > y:#改参数注意点，越小多次传击概率越小
                if x - 0.29 <= s[0] <= x + 0.29 and s[1] > y:
                    # 接近到左右半个冰壶的位置，用这个球作为中介球
                    # if xy[0] - 0.29 <= s[0] <= xy[0] + 0.29 :
                    if xy[0] - 0.29 <= s[0] <= xy[0] + 0.29 :
                        x=s[0]
                        y=s[1]
                    flag=True
                    count=count+1
            if flag and count<=1:
                print('通过传击，击飞对手')
                # return np.array([random.uniform(0.95, 1), (x - 2.375 + 0.03) / 2, 0])  # 速度量什么的可以修改
                '''if x>z[0]:
                    if y>z[1]+5:
                        return np.array([0.97, (x - 2.375 + 0.008) / 2, 0]) 
                    elif y>z[1]+4: 
                        return np.array([0.97, (x - 2.375 + 0.01) / 2, 0]) 
                    else:
                        return np.array([0.97, (x - 2.375 + 0.02) / 2, 0])
                if x<=z[0]:
                    if y>z[1]+5:
                        return np.array([0.97, (x - 2.375 - 0.008) / 2, 0]) 
                    elif y>z[1]+4: 
                        return np.array([0.97, (x - 2.375 - 0.01) / 2, 0]) 
                    else:
                        return np.array([0.97, (x - 2.375 - 0.02) / 2, 0])'''
                '''if x>xy[0]+0.07:
                    if y>z[1]+5:
                        return np.array([0.97, (x - xy[0] + 0.025) / 2, 0]) 
                    elif y>z[1]+4: 
                        return np.array([0.97, (x - xy[0] + 0.03) / 2, 0]) 
                    else:
                        return np.array([0.97, (x - xy[0] + 0.035) / 2, 0])
                elif x<xy[0]-0.07:
                    if y>z[1]+5:
                        return np.array([0.97, (x - xy[0] - 0.025) / 2, 0]) 
                    elif y>z[1]+4: 
                        return np.array([0.97, (x - xy[0] - 0.03) / 2, 0]) 
                    else:
                        return np.array([0.97, (x - xy[0] - 0.035) / 2, 0])
                else:
                    return np.array([0.97, (x - xy[0]) / 2, 0])'''
                if x!=xy[0]:
                    return np.array([0.97, (x + 0.29*(x-xy[0])/dist(xy,[x,y]) - 2.375) / 2, 0])
                else:
                    return np.array([0.97, (x - 2.375)/ 2, 0])
            elif flag and count>1:
                print('做球')
                while True:
                    bias=random.uniform(-0.8, 0.8)
                    if bias<-0.4 or bias>0.4:
                        break
                return np.array([0.97, bias , 0])
            elif state[32]*16!=2 and state[32]*16!=3 and state[32]*16!=4:
                print('直接击飞对手')
                # 选择0.05，这样子0.32-0.145*2=0.03
                return np.array([0.85, (x - 2.375) / 2, 0])
                '''if xy[0]>z[0]:
                    return np.array([0.85, (x - 2.375 - 0.03) / 2, 0])
                elif xy[0]<=z[0]:
                    return np.array([0.85, (x - 2.375 + 0.03) / 2, 0])  # 人工策略3：到此结束'''
        return []
    
    def strategy_fourth(self, state):
        # 人工策略4：护住得分球。比分高的情下启动，得分越高启动概率越高
        score = state[33]
        #if score > 0 and random.uniform(0, 1) < 1:
        if score > 0:             
            houshou = (state[32]*16)%2==0
            xy = getxy(not houshou, state[0:32])
            flag = True
            for i in range(0, 32, 2):
                # 如果有个壶在我方最好壶下方附近，不用保护
                if xy[0] - 0.17 <= state[i] * 4.2996 <= xy[0] + 0.17 and state[i + 1] * 10.4154 >= xy[1] + 2.1:#改参数注意点
                    flag = False
                    break
            if flag:
                action = np.array([-0.75, (xy[0] - 2.375) / 2, 0])  # 人工策略4：到此结束
                print('护住得分球')
                return np.clip(action, -1, 1)
        return []    

    def strategy_forteenth(self, state):        
        #人工策略14：优势局或均势局中路造球
        score = state[33]
        z = [2.375, 4.88]
        s = []
        if score>=0: 
            # 获取对方最好球坐标
            houshou = (state[32]*16)%2==0
            xy = getxy(houshou, state[0:32])
            if len(xy) > 0:
                dis = dist(xy, z)
            else:
                dis = 0.61
            if dis >= 0.61:
                dis = 0.61
            s = [z[0] - dis]
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
            s.append(z[0] + dis)
            s = sorted(s)
            for i in range(0, len(s) - 1):
                if z[0] - dis <= s[i] <= z[0] + dis and z[0] - dis <= s[i + 1] <= z[0] + dis:
                    if s[i + 1] - s[i] >= 0.66:
                        print('中路造球')
                        #return np.array([-0.58, (s[i] + 0.33 - 2.375) / 2, 0])  # 人工策略14结束
                        return np.array([-0.4, (s[i] + 0.33 - 2.375) / 2, 0])  # 人工策略14结束
        return []
    
    def strategy_eighth(self, state):
        # 人工策略8，反侧边隔山打牛球
        score = state[33]
        z = [2.375, 4.88]
        s = []
        # if not (score==0 and state[32]*16<5):
        if score>0 and state[32]*16>4: 
            # 对方的起始坐标 
            start=2
            if (state[32]*16)%2 == 0:
                start=0
            s = []
            for i in range(0, 32, 2):
                s.append(state[i] * 4.2996)
                s.append(state[i + 1] * 10.4154)
            d=[]
            v=[]
            # 遍历对方在划定范围内的壶
            for i in range(start,32,4):
                if z[1]+0.71<=s[i+1]<=z[1]+4.5:#改参数注意点
                    xy=[s[i],s[i+1]]
                    flag=True
                    for j in range(0,32,2):
                        if j==i:
                            continue
                        # 我方壶接近对方壶
                        if xy[0]-0.33<=s[j]<=xy[0]+0.33 and s[j+1]>xy[1]-1 :
                            flag=False
                            break
                    if flag:
                        # 可能是打到对方侧击进圈路线上阻拦
                        d.append((xy[0]-z[0])/2)
                        v.append(getv1(xy[1]))
            if len(d)>0:
                mi=d[0]
                v1=v[0]
                # 偏好于打离中心远的阻挡壶
                for i in range(0,len(d)):
                    if abs(mi)>abs(d[i]):
                        mi=d[i]
                        v1=v[i]
                print('阻挡侧击')
                return np.array([v1-0.05,mi-0.05,0])#人工策略8：结束
        return []

    def strategy_default(self,state):
        # 上述情况以外
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action = self.actor(state).squeeze(0).numpy()
        '''noise = np.random.normal(0, self.policy_noise, size=action.shape)
        action += noise'''
        #print('action:', action)
        print('给定策略之外，网络决策')
        return np.clip(action, -1, 1)

    def update(self, batch_size):
        if len(self.memory) < batch_size:
            return

        self.total_it += 1

        # 从经验回放内存中采样数据
        transitions = self.memory.sample(batch_size)
        batch = Transition(*zip(*transitions))

        states = torch.FloatTensor(np.array(batch.state))
        actions = torch.FloatTensor(np.array(batch.action))
        rewards = torch.FloatTensor(np.array(batch.reward))
        next_states = torch.FloatTensor(np.array(batch.next_state))
        dones = torch.FloatTensor(np.array(batch.done))

        # 更新两个critic网络
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            noise = torch.clamp(torch.randn_like(next_actions) * self.policy_noise, -self.noise_clip,
                                self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, -1, 1)
            Q_targets_next1 = self.target_critic1(next_states, next_actions)
            Q_targets_next2 = self.target_critic2(next_states, next_actions)
            Q_targets_next = torch.min(Q_targets_next1, Q_targets_next2)
            Q_targets = rewards + (1 - dones) * self.gamma * Q_targets_next

        Q = Q_targets.detach().numpy()
        l = []
        for i in range(batch_size):
            l.append(Q[i][i])
        Q_targets = torch.FloatTensor(np.array(l)).unsqueeze(1)

        self.optimizer_critic1.zero_grad()
        Q_expected1 = self.critic1(states, actions)
        critic_loss1 = F.mse_loss(Q_expected1, Q_targets.detach())
        critic_loss1.backward()
        self.optimizer_critic1.step()

        self.optimizer_critic2.zero_grad()
        Q_expected2 = self.critic2(states, actions)
        critic_loss2 = F.mse_loss(Q_expected2, Q_targets.detach())
        critic_loss2.backward()
        self.optimizer_critic2.step()
        print('successful update')

        # 延迟更新策略网络
        if self.total_it % self.policy_delay == 0:
            self.optimizer_actor.zero_grad()
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            actor_loss.backward()
            self.optimizer_actor.step()

            # 软更新target网络
            self.soft_update(self.actor, self.target_actor, self.tau)
            self.soft_update(self.critic1, self.target_critic1, self.tau)
            self.soft_update(self.critic2, self.target_critic2, self.tau)

        if self.ep > self.lim:
            # self.epsa += 1
            self.ep *= self.epsa

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

# 定义经验回放内存
class ReplayMemory:
    # 初始化ReplayMemory类，设置容量为64
    def __init__(self, capacity=64):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    # 将transition添加到memory中
    def push(self, transition):
        # 如果memory的长度小于capacity，则添加一个None
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        # 将transition添加到memory中
        self.memory[self.position] = transition
        # 更新position
        self.position = (self.position + 1) % self.capacity

    # 从memory中随机抽取batch_size个transition
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    # 返回memory的长度
    def __len__(self):
        return len(self.memory)


def getv(x):
    p = [0.210936613469035, 1.91673185127749]
    v = p[0] * x + p[1]
    v1 = 0
    if 2.4 <= v < 2.8:
        v1 = (v - 2.4) / 0.8 - 1
    elif 2.8 <= v < 3.2:
        v1 = (v - 2.8) / 0.4 - 0.5
    elif 3.2 <= v <= 6:
        v1 = (v - 3.2) / 5.6 + 0.5
    return v1

def getv1(x):
    p = [-0.1047, 3.486]
    v = p[0] * x + p[1]
    v1 = 0
    if 2.4 <= v < 2.8:
        v1 = (v - 2.4) / 0.8 - 1
    elif 2.8 <= v < 3.2:
        v1 = (v - 2.8) / 0.4 - 0.5
    elif 3.2 <= v <= 6:
        v1 = (v - 3.2) / 5.6 + 0.5
    return v1

def getv2(xy):
    p = [-0.08411, 0.1947, 0.003559]  # f(x,y) = p00 + p10*x + p01*y
    if xy[0] < 2.375:
        x = 2.375 * 2 - xy[0]
    else:
        x = xy[0]
    v2 = p[0] + p[1] * x + p[2] * xy[1]
    return v2

def dist(xy1, xy2):
    s = 0
    for (a, b) in zip(xy1, xy2):
        s += (a - b) ** 2
    return s ** 0.5

# HOUSE_R0 = 0.15
# HOUSE_R1 = 0.61
# HOUSE_R2 = 1.22
def getp(xy: list, score: float, huihe: float, houshou: bool) -> float:
    z = [2.375, 4.88]
    dis = dist(xy, z)
    p = 0
    if houshou==False:#先手概率被减低了，不想减低的请注释掉这句话
       huihe=0#先手概率被减低了，不想减低的请注释掉这句话
    if dis <= 0.15:
        p = 1
    elif dis <= 0.91:
        p = 1
    else:
        if score < 0:
            p = -3 * score + huihe
        elif score > 0:
            p = 3 * score
    return p

def getb(xy: list, z: list) -> float:
    # xy和中心连线与x轴夹角余弦乘以0.145
    dis = dist(xy, z)
    b = 0.29 * (xy[0] - z[0]) / dis
    return b

def getb1(xy: list, z: list) -> float:
    p1 = [-0.03421, -0.127, 0.1764]  # f(x) = p1*x^2 + p2*x + p3
    p2 = [0.02491, -0.1114, -0.1316]  # f(x) = p1*x^2 + p2*x + p3
    t = (z[1] - xy[1]) / (z[0] - xy[0])
    if t > 0:
        b = p2[0] * t ** 2 + p2[1] * t + p2[2]
    else:
        b = p1[0] * t ** 2 + p1[1] * t + p1[2]
    return b

def getc(xy: list, xy1: list, z: list) -> bool:
    # 当xy1与xy，z三点基本共线时，返回True（但不能就在z点上），否则返回False
    if xy1[0] == z[0] and xy1[1] == z[1]:
        return False
    if xy[0] != z[0]:
        k = (xy[1] - z[1]) / (xy[0] - z[0])
        y1 = k * (xy1[0] - xy[0]) + xy[1] - 0.29
        y2 = k * (xy1[0] - xy[0]) + xy[1] + 0.29
        if y1 <= xy1[1] <= y2:
            return True
        else:
            return False

def getxy(houshou: bool, state: np) -> list:
    # 返回敌人球的坐标中距离中心最近的
    xy = []
    z = [2.375, 4.88]
    # r = 1.830
    r = 1.975
    re = []
    if houshou:
        for i in range(0, 32, 4):
            xy = [state[i] * 4.2996, state[i + 1] * 10.4154]
            dis = dist(xy, z)
            if dis < r:
                r = dis
                re = [state[i] * 4.2996, state[i + 1] * 10.4154]
    else:
        for i in range(2, 32, 4):
            xy = [state[i] * 4.2996, state[i + 1] * 10.4154]
            dis = dist(xy, z)
            if dis < r:
                r = dis
                re = [state[i] * 4.2996, state[i + 1] * 10.4154]
    return re

# '4.3996', '10.4154'
def huan(action1):
    action = []
    for i in range(0, 32, 2):
        action.append(action1[i] / 4.2996)
        action.append(action1[i + 1] / 10.4154)
    return np.array(action)

def huan1(value: float, mode: int):
    v1 = 3
    v2 = 0
    v3 = 0
    if mode == 0:
        if value >= -1 and value < -0.5:
            v1 = 0.8 * (value + 1) + 2.4
        elif value >= -0.5 and value < 0.5:
            v1 = 0.4 * (value + 0.5) + 2.8
        elif value >= 0.5 and value <= 1:
            v1 = 5.6 * (value - 0.5) + 3.2
        return v1
    elif mode == 1:
        v2 = 2 * (value + 1) - 2
        return v2
    elif mode == 2:
        v3 = 3 * (value + 1) - 3
        return v3

# 策略
def strategy(action):
    # print('请输入速度，偏移角度，角速度（以空格间隔）：')
    # text = input()
    lis = []
    lis.append(huan1(action[0], 0))
    lis.append(huan1(action[1], 1))
    lis.append(huan1(action[2], 2))
    text = " ".join(list(map(str, lis)))
    bestshot = str("BESTSHOT ")
    bestshot = bestshot + text
    return bestshot

def get_reward(houshou: bool, state: np) -> list:
    # 初始化变量
    xy1 = []
    xy2 = []
    dis1 = []
    dis2 = []
    s = 0
    z = [2.375, 4.88]
    # r = 1.830
    r = 1.975
    num1 = 0
    num2 = 0
    re = []
    # 遍历state，将xy1和xy2分别存入xy1和xy2列表中
    for i in range(0, 32, 4):
        xy1.append([state[i], state[i + 1]])
        if state[i] != 0 and state[i + 1] != 0:
            num1 += 1
        xy2.append([state[i + 2], state[i + 3]])
        if state[i + 2] != 0 and state[i + 3] != 0:
            num2 += 1
    # 计算xy1和xy2与z的距离，并存入dis1和dis2列表中
    for i in range(8):
        dis1.append(dist(xy1[i], z))
        dis2.append(dist(xy2[i], z))
    # 对dis1和dis2进行排序
    dis1 = sorted(dis1)
    dis2 = sorted(dis2)
    # 判断双方距离中心最近的壶是否都在圈外
    if dis1[0] > r and dis2[0] > r: 
        if houshou:
            re.append(-s / 8)
            re.append(num1 / 8)
            return re
        else:
            re.append(s / 8)
            re.append(num2 / 8)
            return re
    # 判断先手和后手距离中心最近的壶的距离
    elif dis1[0] < dis2[0]:
        if dis2[0] < r:
            r = dis2[0]     #后手有壶在圈内，缩圈
        for i in range(8):
            if dis1[i] < r:
                s += 1      #先手能够得到的分数（比对面还要近的壶数）
            else:
                break
    else:
        if dis1[0] < r:
            r = dis1[0]
        for i in range(8):
            if dis2[i] < r:
                s -= 1
            else:
                break
    # 根据houshou的值，返回不同的re列表
    if houshou:
        re.append(-s / 8)
        re.append(num1 / 8)
        return re
    else:
        re.append(s / 8)
        re.append(num2 / 8)
        return re

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Description of your program')
    parser.add_argument('-p', '--port', help='tcp server port', default="7788", required=False)
    parser.add_argument('-host', '--host', help='host', default="127.0.0.1", required=False)
    args, unknown = parser.parse_known_args()

    # 连接host(无需修改)
    host = args.host
    # 默认连接端口(无需修改)
    port = int(args.port)

    obj = socket.socket()
    obj.connect((host, port))

    retNullTime = 0
    while True:
        ret = recv_message(obj)
        messageList = ret.split(" ")
        if ret == "":
            retNullTime = retNullTime + 1
        if retNullTime == 20:
            break
        if messageList[0] == "NAME":
            order = messageList[1]
        if messageList[0] == "ISREADY":
            time.sleep(0.5)
            send_message(obj, "READYOK")
            time.sleep(0.5)
            send_message(obj, "NAME XMU自动化")
            break
    # 定义经验回放的数据结构
    Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
    noise = 0.1
    agent = TD3Agent(state_dim=35, action_dim=3, hidden_dim=256, gamma=0.9, tau=0.0005, lr_actor=0.0005, lr_critic=0.0005,
                 policy_delay=8, policy_noise=noise, noise_clip=2 * noise)
    state1 = []
    state2 = []
    action1 = []
    batch_size = 32
    total_size = 1
    startflag1 = False
    startflag2 = False
    startflag3 = False
    if total_size != 0:
        agent.actor.load_state_dict(torch.load("actor3.pth"))
        agent.target_actor.load_state_dict(torch.load("target_actor3.pth"))
        agent.critic1.load_state_dict(torch.load("critic13.pth"))
        agent.target_critic1.load_state_dict(torch.load("target_critic13.pth"))
        agent.critic2.load_state_dict(torch.load("critic23.pth"))
        agent.target_critic2.load_state_dict(torch.load("target_critic23.pth"))
        print('successful load')
    else:
        agent.save_models()
    agent.actor
    # 初始化
    shotnum = str("0")
    order = str("Player1")  # 先后手
    # 每一轮比赛当前完成投掷数
    state = []
    count = 0
    com = 0
    retNullTime = 0
    houshou = 0
    shotnum1 = []

    # 无限循环
    while True:
        # 接收消息
        ret = recv_message(obj)
        # 将消息按空格分割成列表
        messageList = ret.split(" ")
        # 如果接收到的消息为空，则将retNullTime加1
        if ret == "":
            retNullTime = retNullTime + 1
        # 如果retNullTime等于5，则跳出循环
        if retNullTime == 20:
            break
        # 如果消息列表的第一个元素为"NAME"，则将order赋值为消息列表的第二个元素
        if messageList[0] == "NAME":
            order = messageList[1]
        # 如果消息列表的第一个元素为"ISREADY"，则先休眠0.5秒，然后发送"READYOK"，再休眠0.5秒，然后发送"NAME XMU自动化"
        if messageList[0] == "ISREADY":
            time.sleep(0.5)
            send_message(obj, "READYOK")
            time.sleep(0.5)
            send_message(obj, "NAME XMU自动化")
        # 如果消息列表的第一个元素为"POSITION"，则将state赋值为消息列表的第二个元素到第33个元素
        if messageList[0] == "POSITION":
            if state:
                state = []
            state.append(ret.split(" ")[1:33])
            # 如果startflag1为True，则将state[0]转换为浮点数，并添加到state2中，然后将startflag1赋值为False
            if startflag1:
                s = np.array(list(map(float, state[0])))
                state2.append(s)
                startflag1 = False
        # 如果消息列表的第一个元素为"SETSTATE"，则将shotnum赋值为消息列表的第二个元素
        if messageList[0] == "SETSTATE":
            # 当前完成投掷数
            shotnum = ret.split(" ")[1]
            state.append(int(shotnum)+1)
        if messageList[0] == "GO":
            # 将state[0]转换为浮点数数组，并添加到state1中
            s = np.array(list(map(float, state[0])))
            state1.append(s)
            # 设置startflag1为True
            startflag1 = True
            # 将s和state[1]转换为浮点数，并添加到state3中
            state3 = np.append(huan(s), int(state[1]) / 16)
            # 如果state[1]是偶数，则调用get_reward函数，并传入False和s
            # 如果后手，则调用get_reward函数，并传入False和s
            # if int(state[1]) % 2 == 0:
            #     re = get_reward(False, s)
            # # 如果先手，则调用get_reward函数，并传入True和s
            # else:
            #     re = get_reward(True, s)
            houshou = int(state[1]) % 2 == 0
            re = get_reward(houshou, s)
            # 将re[0]和re[1]添加到state3中
            state3 = np.append(state3, re[0])
            state3 = np.append(state3, re[1])
            # 调用agent的select_action函数，并传入state3，获取action
            action = agent.select_action(state3)
            '''这个地方不知道有没有问题'''
            if action is None:
                print("No action returned from select_action. Using default action.")
                action = action = np.random.uniform(-0.99, 0.99, 3)  # 假设动作维度为3，根据需要调整
            # 将action添加到action1中
            action1.append(np.array(action))
            # 调用strategy函数，并传入action，获取shot
            shot = strategy(action)
            # 调用send_message函数，并传入obj和shot
            send_message(obj, shot)
            # 将com设置为0
            com = 0
            # 将state[1]转换为整数，并赋值给episode
            episode = int(state[1])
            # 将state[1]除以16的结果添加到shotnum1中
            shotnum1.append(int(state[1]) / 16)
        if messageList[0] == "SCORE" and com != 1:
            # SCORE = float(messageList[1])
            # if SCORE!=0:
            # reward1=0
            reward1 = []
            reward2 = []
            if episode == 16:
                startflag1 = False
                startflag3 = True
                s = np.array(list(map(float, state[0])))
                state2.append(s)
                state3 = np.array([])
            for i in range(0, 7):
                reward1 = get_reward(startflag3, state1[i])
                reward2 = get_reward(startflag3, state2[i])
                state3 = np.append(huan(state1[i]), shotnum1[i])
                state3 = np.append(state3, reward1[0])
                state3 = np.append(state3, reward1[1])
                state4 = np.append(huan(state2[i]), (shotnum1[i] * 16 + 1) / 16)
                state4 = np.append(state4, reward2[0])
                state4 = np.append(state4, reward2[1])
                transition = Transition(state=state3, action=action1[i],
                                        reward=reward2[0],
                                        next_state=state4, done=False)
                agent.memory.push(transition)
            if startflag3:
                reward1 = get_reward(startflag3, state1[7])
                reward2 = get_reward(startflag3, state2[7])
                state3 = np.append(huan(state1[7]), shotnum1[7])
                state3 = np.append(state3, reward1[0])
                state3 = np.append(state3, reward1[1])
                state4 = np.append(huan(state2[7]), (shotnum1[7] * 16 + 1) / 16)
                state4 = np.append(state4, reward2[0])
                state4 = np.append(state4, reward2[1])
                transition = Transition(state=state3, action=action1[7],
                                        reward=reward2[0],
                                        next_state=state4, done=True)
                agent.memory.push(transition)
                startflag3 = False

            else:
                reward1 = get_reward(startflag3, state1[7])
                reward2 = get_reward(startflag3, state2[7])
                state3 = np.append(huan(state1[7]), shotnum1[7])
                state3 = np.append(state3, reward1[0])
                state3 = np.append(state3, reward1[1])
                state4 = np.append(huan(state2[7]), (shotnum1[7] * 16 + 1) / 16)
                state4 = np.append(state4, reward2[0])
                state4 = np.append(state4, reward2[1])
                transition = Transition(state=state3, action=action1[7],
                                        reward=reward2[0],
                                        next_state=state4, done=False)
                agent.memory.push(transition)
            agent.update(batch_size=batch_size)
            #print('successful update')
            state1 = []
            state2 = []
            action1 = []
            shotnum1 = []
            total_size += 1
            # agent.flag=0
            print('当前轮数：%d' % (total_size))
            com = 1

            if total_size % 20 == 0 and total_size != 0:
                agent.save_models()
            if total_size == 1001:
                break

        if messageList[0] == "MOTIONINFO":
            x_coordinate = float(messageList[1])
            y_coordinate = float(messageList[2])
            x_velocity = float(messageList[3])
            y_velocity = float(messageList[4])
            angular_velocity = float(messageList[5])