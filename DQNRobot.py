# -*- coding: utf-8 -*-

import os, time, math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import multiprocessing as mp
import torch.multiprocessing as tmp
from AIRobot import AIRobot

BATCH_SIZE = 32  # 批次尺寸
LR = 0.0001  # 学习率
EPSILON = 0.7  # 最优选择动作百分比
GAMMA = 0.9  # 奖励折扣因子
TARGET_REPLACE_ITER = 500  # Q现实网络的更新频率
MEMORY_CAPACITY = 10000  # 记忆库大小

# 获取某一冰壶距离营垒圆心的距离
def get_dist(x, y):
    House_x = 2.375
    House_y = 4.88
    return math.sqrt((x - House_x) ** 2 + (y - House_y) ** 2)

# 根据冰壶比赛服务器发送来的场上冰壶位置坐标列表获取得分情况并生成信息状态数组
def get_infostate(position):
    House_R = 1.830
    Stone_R = 0.145

    init = np.empty([8], dtype=float)
    gote = np.empty([8], dtype=float)
    both = np.empty([16], dtype=float)
    # 计算双方冰壶到营垒圆心的距离
    for i in range(8):
        init[i] = get_dist(position[4 * i], position[4 * i + 1])
        both[2 * i] = init[i]
        gote[i] = get_dist(position[4 * i + 2], position[4 * i + 3])
        both[2 * i + 1] = gote[i]
    # 找到距离圆心较远一方距离圆心最近的壶
    if min(init) <= min(gote):
        win = 0  # 先手得分
        d_std = min(gote)
    else:
        win = 1  # 后手得分
        d_std = min(init)

    infostate = []  # 状态数组
    init_score = 0  # 先手得分
    # 16个冰壶依次处理
    for i in range(16):
        x = position[2 * i]  # x坐标
        y = position[2 * i + 1]  # y坐标
        dist = both[i]  # 到营垒圆心的距离
        sn = i % 2 + 1  # 投掷顺序
        if (dist < d_std) and (dist < (House_R + Stone_R)) and ((i % 2) == win):
            valid = 1  # 是有效得分壶
            # 如果是先手得分
            if win == 0:
                init_score = init_score + 1
            # 如果是后手得分
            else:
                init_score = init_score - 1
        else:
            valid = 0  # 不是有效得分壶
        # 仅添加有效壶
        if x != 0 or y != 0:
            infostate.append([x, y, dist, sn, valid])
    # 按dist升序排列
    infostate = sorted(infostate, key=lambda x: x[2])

    # 无效壶补0
    for i in range(16 - len(infostate)):
        infostate.append([0, 0, 0, 0, 0])

    # 返回先手得分和转为一维的状态数组
    return init_score, np.array(infostate).flatten()

#低速：在(2.4,2.7)之间以0.1为步长进行离散
slow = np.arange(2.4, 2.7, 0.1)
#中速：在(2.8,3.2)之间以0.05为步长进行离散
normal = np.arange(2.8, 3.2, 0.05)
#高速
fast = np.array([4,5,6])
#将低速、中速、高速三个数组连接起来
speed = np.concatenate((slow, normal, fast))

#横向偏移在(-2,2)之间以0.4为步长进行离散
deviation = np.arange(-2, 2, 0.4)
#角速度在(-3.14, 3.14)之间以0.628为步长进行离散
angspeed = np.arange(-3.14, 3.14, 0.628)

n = 0
#初始化动作列表
action_list = np.empty([1600, 3], dtype=float)
#遍历速度、横向偏移、角速度组合成各种动作
for i in speed:
    for j in deviation:
        for k in angspeed:
            action_list[n,] = [i, j, k]
            n += 1

class Net(nn.Module):
    # 初始化网络
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(80, 256)  # 定义全连接层1
        self.fc1.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重
        self.fc2 = nn.Linear(256, 1024)  # 定义全连接层2
        self.fc2.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重
        self.out = nn.Linear(1024, 1600)  # 定义输出层
        self.out.weight.data.normal_(0, 0.1)  # 按(0, 0.1)的正态分布初始化权重

    # 网络前向推理
    def forward(self, x):
        x = self.fc1(x)  # 输入张量经全连接层1传递
        x = F.relu(x)  # 经relu函数激活
        x = self.fc2(x)  # 经全连接层2传递
        x = F.relu(x)  # 经relu函数激活
        return self.out(x)  # 经输出层传递得到输出张量

class DQN(object):
    def __init__(self):
        self.eval_net = Net()  # 初始化评价网络
        self.target_net = Net()  # 初始化目标网络
        self.sum_loss = 0  # 初始化loss值
        self.learn_step_counter = 0  # 用于目标网络更新计时
        self.memory_counter = 0  # 记忆库计数
        self.memory = np.zeros((MEMORY_CAPACITY, 80 * 2 + 2))  # 初始化记忆库
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # 设定torch的优化器为Adam
        self.loss_func = nn.MSELoss()  # 以均方误差作为loss值
        self.min_loss = 10000

    # 根据输入状态x返回输出动作的索引（而不是动作）
    def choose_action(self, x):
        # 选最优动作
        if np.random.uniform() < EPSILON:
            x = Variable(torch.FloatTensor(x))  # 将x转为pytorch变量 shape-torch.Size([80])
            actions_eval = self.eval_net(x)  # 评价网络前向推理 shape-torch.Size([1600])
            action = int(actions_eval.max(0)[1])  # 返回概率最大的动作索引
        # 选随机动作
        else:
            action = np.random.randint(0, 1600)  # 在0-1600之间选一个随机整数
        return action

        # 存储经验数据（s是输入状态，a是输出动作，r是奖励，s_是下一刻的状态）

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))  # 将输入元组的元素数组按水平方向进行叠加
        # 如果记忆库满了, 就覆盖老数据
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    # 学习经验数据
    def learn(self):
        # 每隔TARGET_REPLACE_ITER次更新目标网络参数
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # 抽取记忆库中的批数据
        size = min(self.memory_counter, MEMORY_CAPACITY)
        sample_index = np.random.choice(size, BATCH_SIZE)
        b_memory = self.memory[sample_index, :]  # 抽取出来的数据 shape-(32, 162)
        b_s = Variable(torch.FloatTensor(b_memory[:, :80]))  # 输入数据的状态 shape-torch.Size([32, 80])
        b_a = Variable(torch.LongTensor(b_memory[:, 80:81]))  # 输入数据的动作 shape-torch.Size([32, 1])
        b_r = Variable(torch.FloatTensor(b_memory[:, 81:82]))  # 输入数据的奖励 shape-torch.Size([32, 1])
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -80:]))  # 输入数据的下一个状态 shape-torch.Size([32, 80])

        # 针对做过的动作 b_a 来选 q_eval 的值
        self.eval_net.train()  # 设定当前处于训练模式
        actions_eval = self.eval_net(b_s)  # 评价网络前向推理 shape-torch.Size([32, 1600])
        q_eval = actions_eval.gather(1, b_a)  # 选取第1维第b_a个数为评估Q值 shape-torch.Size([32, 1])

        max_next_q_values = torch.zeros(32, dtype=torch.float).unsqueeze(dim=1)  # shape-torch.Size([32, 1])
        for i in range(BATCH_SIZE):
            action_target = self.target_net(b_s_[i]).detach()  # 目标网络前向推理 shape-torch.Size([1600])
            max_next_q_values[i] = float(action_target.max(0)[0])  # 返回输出张量中的最大值
        q_target = (b_r + GAMMA * max_next_q_values)  # 计算目标Q值 shape-torch.Size([32, 1])

        # 计算loss值
        loss = self.loss_func(q_eval, q_target)
        loss_item = loss.item()
        if loss_item < self.min_loss:
            self.min_loss = loss_item

        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 将loss进行反向传播并计算网络参数的梯度
        self.optimizer.step()  # 优化器进行更新
        return loss_item

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8, weight_decay=0):
        super(SharedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class ParallelDQN:
    def __init__(self, ports):
        self.ports = ports
        self.ctx = mp.get_context('spawn')
        self.shared_memory = mp.Queue(maxsize=2000)

        self.global_dqn_init = DQN()
        self.global_dqn_init.eval_net.share_memory()
        self.global_dqn_init.target_net.share_memory()
        self.optimizer_init = SharedAdam(self.global_dqn_init.eval_net.parameters(), lr=LR)

        self.global_dqn_dote = DQN()
        self.global_dqn_dote.eval_net.share_memory()
        self.global_dqn_dote.target_net.share_memory()
        self.optimizer_dote = SharedAdam(self.global_dqn_dote.eval_net.parameters(), lr=LR)

        self.init_model_file = 'model/DQN_init_parallel.pth'
        self.dote_model_file = 'model/DQN_dote_parallel.pth'

        # 加载已有的模型（如果存在）
        if os.path.exists(self.init_model_file):
            self.global_dqn_init.eval_net.load_state_dict(torch.load(self.init_model_file))
            self.global_dqn_init.target_net.load_state_dict(torch.load(self.init_model_file))
        if os.path.exists(self.dote_model_file):
            self.global_dqn_dote.eval_net.load_state_dict(torch.load(self.dote_model_file))
            self.global_dqn_dote.target_net.load_state_dict(torch.load(self.dote_model_file))
        # 日志文件
        self.log_file_name = 'log/DQN_' + time.strftime("%y%m%d_%H%M%S") + '.log'

    def train(self):
        processes = []
        for i in range(len(self.ports)):
            p = mp.Process(target=self.agent_process, args=(i, self.ports[i]))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()

    def agent_process(self, agent_id, port):
        agent = ParallelDQNRobot(agent_id, port, self.shared_memory, self.log_file_name,
                                 self.global_dqn_init, self.global_dqn_dote,
                                 self.optimizer_init, self.optimizer_dote,
                                 self.init_model_file, self.dote_model_file)
        agent.run()

class ParallelDQNRobot(AIRobot):
    def __init__(self, agent_id, port, shared_memory, log_file_name, global_dqn_init, global_dqn_dote,
                 optimizer_init, optimizer_dote, init_model_file, dote_model_file):
        super().__init__(key="local", name=f"Agent_{agent_id}", host="127.0.0.1", port=port)

        self.agent_id = agent_id
        self.shared_memory = shared_memory
        self.global_dqn_init = global_dqn_init
        self.global_dqn_dote = global_dqn_dote
        self.optimizer_init = optimizer_init
        self.optimizer_dote = optimizer_dote
        self.init_model_file = init_model_file
        self.dote_model_file = dote_model_file
        self.log_file_name = log_file_name
        self.learn_start = 100  # 学习起始局数
        self.round_max = 10000  # 最大训练局数

        # 创建本地DQN
        self.dqn_init = DQN()
        self.dqn_dote = DQN()
        self.sync_with_global()

    # 根据当前比分获取奖励分数
    def get_reward(self, this_score):
        House_R = 1.830
        Stone_R = 0.145
        reward = this_score - self.last_score
        if (reward == 0):
            x = self.position[2 * self.shot_num]
            y = self.position[2 * self.shot_num + 1]
            dist = self.get_dist(x, y)
            if dist < (House_R + Stone_R):
                reward = 1 - dist / (House_R + Stone_R)
        return reward

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        if not self.shared_memory.full():
            self.shared_memory.put(transition)

    def learn(self):
        if self.shared_memory.qsize() < BATCH_SIZE:
            return 0

        batch = [self.shared_memory.get() for _ in range(BATCH_SIZE)]
        b_memory = np.array(batch)
        b_s = torch.FloatTensor(b_memory[:, :80])
        b_a = torch.LongTensor(b_memory[:, 80:81].astype(int))
        b_r = torch.FloatTensor(b_memory[:, 81:82])
        b_s_ = torch.FloatTensor(b_memory[:, -80:])

        if self.player_is_init:
            dqn = self.dqn_init
            global_dqn = self.global_dqn_init
            optimizer = self.optimizer_init
        else:
            dqn = self.dqn_dote
            global_dqn = self.global_dqn_dote
            optimizer = self.optimizer_dote

        # 使用本地网络计算损失
        q_eval = dqn.eval_net(b_s).gather(1, b_a)
        q_next = dqn.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = dqn.loss_func(q_eval, q_target)

        # 计算梯度
        optimizer.zero_grad()
        loss.backward()
        # 将本地梯度应用到全局网络
        for local_param, global_param in zip(dqn.eval_net.parameters(),
                                             global_dqn.eval_net.parameters()):
            if global_param.grad is not None:
                global_param._grad = local_param.grad

        # 更新全局网络
        optimizer.step()
        return loss.item()

    def sync_with_global(self):
        self.dqn_init.eval_net.load_state_dict(self.global_dqn_init.eval_net.state_dict())
        self.dqn_init.target_net.load_state_dict(self.global_dqn_init.target_net.state_dict())
        self.dqn_dote.eval_net.load_state_dict(self.global_dqn_dote.eval_net.state_dict())
        self.dqn_dote.target_net.load_state_dict(self.global_dqn_dote.target_net.state_dict())

    def recv_setstate(self, msg_list):
        #当前完成投掷数
        self.shot_num = int(msg_list[0])
        #总对局数
        self.round_total = int(msg_list[2])

        #达到最大局数则退出训练
        if self.round_num == self.round_max:
            self.on_line = False
            return

        #每一局开始时将历史比分清零
        if (self.shot_num == 0):
            self.last_score = 0
            #根据先后手选取模型并设定当前选手第一壶是当局比赛的第几壶
            if self.player_is_init:
                self.dqn = self.dqn_init
                self.first_shot = 0
            else:
                self.dqn = self.dqn_dote
                self.first_shot = 1
        this_score = 0

        #当前选手第1壶投出前
        if self.shot_num == self.first_shot:
            init_score, self.s1 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s1)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第1壶投出后
        elif self.shot_num == self.first_shot + 1:
            init_score, s1_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s1, self.A, reward, s1_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第2壶投出前
        elif self.shot_num == self.first_shot + 2:
            init_score, self.s2 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s2)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第2壶投出后
        elif self.shot_num == self.first_shot + 3:
            init_score, s2_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s2, self.A, reward, s2_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第3壶投出前
        elif self.shot_num == self.first_shot + 4:
            init_score, self.s3 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s3)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第3壶投出后
        elif self.shot_num == self.first_shot + 5:
            init_score, s3_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s3, self.A, reward, s3_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第4壶投出前
        elif self.shot_num == self.first_shot + 6:
            init_score, self.s4 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s4)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第4壶投出后
        elif self.shot_num == self.first_shot + 7:
            init_score, s4_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s4, self.A, reward, s4_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第5壶投出前
        elif self.shot_num == self.first_shot + 8:
            init_score, self.s5 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s5)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第5壶投出后
        elif self.shot_num == self.first_shot + 9:
            init_score, s5_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s5, self.A, reward, s5_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第6壶投出前
        elif self.shot_num == self.first_shot + 10:
            init_score, self.s6 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s6)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第6壶投出后
        elif self.shot_num == self.first_shot + 11:
            init_score, s6_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s6, self.A, reward, s6_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第7壶投出前
        elif self.shot_num == self.first_shot + 12:
            init_score, self.s7 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s7)
            self.action = action_list[self.A]
            self.last_score = (1 - 2 * self.first_shot) * init_score
        #当前选手第7壶投出后
        elif self.shot_num == self.first_shot + 13:
            init_score, s7_ = get_infostate(self.position)
            this_score = (1 - 2 * self.first_shot) * init_score
            reward = self.get_reward(this_score)
            self.store_transition(self.s7, self.A, reward, s7_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()
        #当前选手第8壶投出前
        elif self.shot_num == self.first_shot + 14:
            _, self.s8 = get_infostate(self.position)
            self.A = self.dqn.choose_action(self.s8)
            self.action = action_list[self.A]
        #当前选手第8壶投出后
        elif self.shot_num == self.first_shot + 15:
            _, self.s8_ = get_infostate(self.position)

        if self.shot_num == 16:
            if self.score > 0:
                reward = 5 * self.score
            else:
                reward = 0
            self.store_transition(self.s8, self.A, reward, self.s8_)
            if self.dqn.memory_counter > self.learn_start:
                loss = self.learn()

            self.round_num += 1
            log_file = open(self.log_file_name, 'a+')
            log_file.write(f"Agent {self.agent_id} - score {self.score} {self.round_num}\n")
            if self.dqn.memory_counter > self.learn_start:
                log_file.write(f"Agent {self.agent_id} - loss {loss} {self.round_num}\n")
            log_file.close()

            if self.round_num % 50 == 0:
                if self.player_is_init:
                    net_params = self.global_dqn_init.eval_net.state_dict()
                    torch.save(net_params, self.init_model_file)
                else:
                    net_params = self.global_dqn_dote.eval_net.state_dict()
                    torch.save(net_params, self.dote_model_file)
                print(f'Agent {self.agent_id}: Checkpoint Saved')
        self.sync_with_global()

    def get_bestshot(self):
        return "BESTSHOT " + str(self.action)[1:-1].replace(',', '')

    def run(self):
        self.recv_forever()


if __name__ == "__main__":
    # 定义端口列表
    #ports = [7788, 7789]  # 为每个agent指定一个端口
    ports = [7788]
    parallel_dqn = ParallelDQN(ports=ports)
    parallel_dqn.train()