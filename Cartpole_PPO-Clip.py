# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 21:38:03 2024

@author: zhaodf, zhangtc
"""
#CartPole环境具有一个放置在小车上的平衡杆,智能体需要学习如何在垂直方向保持杆的平衡,整个过程中小车在不断运动
#智能体可获取小车的位置,速度,杆的角度和角速度,并可以在小车的任一端(左、右)施加作用力
#开始时杆直立,每一个保持杆直立的step奖励+1,若杆偏离垂直位置大于12度或小车移出视窗则回合结束
#每个回合至多500 steps

#详细介绍：https://gymnasium.farama.org/environments/classic_control/cart_pole/

#更新了Python3.9和tf2.10

import matplotlib
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import gym
import copy
matplotlib.use('TkAgg')

# 创建Actor网络（演员、策略网络）
def build_actor_network(state_dim, action_dim):
    # 使用 Keras Sequential 模型定义一个简单的前馈神经网络
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=action_dim, activation='softmax')  # 添加输出层，节点数等于动作空间的维度，使用softmax激活函数以输出概率分布
    ])
    model.build(input_shape=(None, state_dim))  # 构建模型时指定输入形状为 (None, state_dim)，None 表示可以接受任意批次大小
    return model

# 创建Critic网络（评论家、价值网络）
def build_critic_network(state_dim):
    # 使用 Keras Sequential 模型定义一个简单的前馈神经网络
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')  # 添加输出层，只有一个节点，使用线性激活函数以输出状态的价值
    ])
    model.build(input_shape=(None, state_dim))  # 构建模型时指定输入形状为 (None, state_dim)
    return model

# 定义 Actor 类
class Actor(object):
    def __init__(self, state_dim, action_dim, lr):
        self.action_dim = action_dim  # 动作空间维度
        self.old_policy = build_actor_network(state_dim, action_dim)  # 创建旧策略网络
        self.new_policy = build_actor_network(state_dim, action_dim)  # 创建新策略网络
        self.update_policy()  # 初始化时将旧策略更新为新策略
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # 创建 Adam 优化器用于训练新策略网络

    # 根据当前状态选择动作
    def choice_action(self, in_state):
        # 使用旧策略网络计算当前状态下的动作概率分布，停止梯度以确保在训练时不会更新这个网络
        local_policy = tf.stop_gradient(self.old_policy(
            np.array([in_state])
        )).numpy()[0]  # 获取概率分布的 NumPy 数组并选择第一个（也是唯一的）元素
        # 从动作概率分布中随机选择一个动作
        return np.random.choice(
            self.action_dim,  # 动作空间的维度
            p=local_policy  # 动作的概率分布
        ), local_policy  # 返回选择的动作和策略概率分布

    # 将新策略网络的权重赋值给旧策略网络
    def update_policy(self):
        self.old_policy.set_weights(
            self.new_policy.get_weights()  # 将新策略网络的权重复制到旧策略网络
        )

    # 更新新策略网络
    def learn(self, batch_state, batch_action, le_advantage, epsilon=0.2):
        le_advantage = np.reshape(le_advantage, newshape=(-1))  # 将优势函数数组重塑为一维数组
        # 创建动作索引，用于从策略网络中提取对应的动作概率
        batch_action = tf.stack([tf.range(tf.shape(batch_action)[0], dtype=tf.int32), batch_action], axis=1)
        old_policy = self.old_policy(batch_state)  # 计算旧策略网络下的动作概率
        with tf.GradientTape() as tape:
            new_policy = self.new_policy(batch_state)  # 计算新策略网络下的动作概率
            # 从新旧策略中提取对应动作的概率
            pi_prob = tf.gather_nd(params=new_policy, indices=batch_action)
            old_policy_prob = tf.gather_nd(params=old_policy, indices=batch_action)
            ratio = pi_prob / (old_policy_prob + 1e-6)  # 计算概率比（重要度采样比）
            surr1 = ratio * le_advantage  # 计算目标函数的第一个部分
            surr2 = tf.clip_by_value(ratio, clip_value_min=1.0 - epsilon, clip_value_max=1.0 + epsilon) * le_advantage
            # 计算目标函数的第二部分，进行裁剪以稳定训练
            loss = - tf.reduce_mean(tf.minimum(surr1, surr2))  # 计算损失函数，取两个部分中的较小值的均值并取负值（因为我们最小化损失）
        grad = tape.gradient(loss, self.new_policy.trainable_variables)  # 计算损失函数对新策略网络权重的梯度
        self.optimizer.apply_gradients(zip(grad, self.new_policy.trainable_variables))  # 应用梯度更新新策略网络的权重

# 定义 Critic 类
class Critic(object):
    def __init__(self, state_dim, lr):
        self.value = build_critic_network(state_dim)  # 创建价值网络
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr)  # 创建 Adam 优化器用于训练价值网络

    # 计算优势函数
    def get_advantage(self, get_state, get_reward):
        return get_reward - self.value.predict(get_state, verbose=0)  # 计算优势函数，当前奖励减去预测的状态价值

    # 计算状态的价值
    def get_value(self, input_state):
        return self.value.predict(
            input_state,
            verbose=0  # 不显示预测的详细信息
        )  # 预测给定状态的价值

    # 更新价值网络
    def learn(self, batch_state, batch_reward):
        with tf.GradientTape() as tape:
            value_predict = self.value(batch_state)  # 计算价值网络的预测
            loss = tf.keras.losses.mean_squared_error(batch_reward, value_predict)  # 计算均方误差损失
        grad = tape.gradient(loss, self.value.trainable_variables)  # 计算损失函数对价值网络权重的梯度
        self.optimizer.apply_gradients(zip(grad, self.value.trainable_variables))  # 应用梯度更新价值网络的权重


# 主程序
if __name__ == '__main__':
    episodes = 200  # 训练的总回合数
    env = gym.make("CartPole-v1", render_mode="human")  # 创建 CartPole 环境
    A_learning_rate = 1e-3  # Actor 网络的学习率
    C_learning_rate = 1e-3  # Critic 网络的学习率
    actor = Actor(4, 2, A_learning_rate)  # 创建 Actor 对象，状态维度为 4，动作维度为 2，学习率为 1e-3
    critic = Critic(4, C_learning_rate)  # 创建 Critic 对象，状态维度为 4，学习率为 1e-3
    gamma = 0.9  # 折扣因子，决定未来奖励的权重
    lam = 0.98  # Generalized Advantage Estimation (GAE) 的 λ，用于平衡偏差和方差
    assert 0.0 <= lam <= 1.0  # 确保 λ 在合法范围内,λ 必须介于(0,1)
    K_epoch = 10  # 每个回合的训练轮数
    assert K_epoch > 1  # 确保 K_epoch 大于 1，K_epoch必须大于1,不然计算的重要性采样没有意义

    plot_score = []  # 用于记录每个回合的得分
    for e in range(episodes):  # 进行训练的每一个回合
        state = env.reset()  # 重置环境，获取初始状态
        S, A, R, nS = [], [], [], []  # 初始化记录状态、动作、奖励和下一个状态的列表
        score = 0.0  # 当前回合的总得分
        state = state[0]
        print("state:", state)
        while True:  # 循环直到回合结束
            action, policy = actor.choice_action(state)  # 根据当前状态选择一个动作
            next_state, reward, terminated, turncated, _ = env.step(action)  # 执行动作，获取下一个状态、奖励和是否结束标志
            done = terminated or turncated # 终止条件
            score += reward  # 累加奖励以计算总得分
            S.append(state)  # 记录当前状态
            A.append(action)  # 记录选择的动作
            R.append(reward)  # 记录获得的奖励
            nS.append(next_state)  # 记录下一个状态
            state = copy.deepcopy(next_state)  # 更新状态，使用深拷贝以防止状态被意外修改
            if done:  # 如果回合结束
                discounted_r = []  # 用于存储折扣奖励的列表
                tmp_r = 0.0  # 临时变量，用于计算折扣奖励
                v_nS = critic.get_value(np.array(nS, dtype=np.float64))  # 获取下一个状态的价值估计
                v_nS[-1] = 0  # 将最后一个状态的价值设为 0
                for r, vs in zip(R[::-1], v_nS[::-1]):  # 从后向前计算折扣奖励
                    tmp_r = r + gamma * (lam * tmp_r + (1 - lam) * vs[0])  # 计算折扣奖励
                    discounted_r.append(np.array([tmp_r]))  # 将奖励加入列表
                discounted_r.reverse()  # 反转奖励列表以恢复正确的顺序

                bs = np.array(S, dtype=np.float64)  # 将状态列表转换为 NumPy 数组
                ba = np.array(A)  # 将动作列表转换为 NumPy 数组
                br = np.array(discounted_r, dtype=np.float64)  # 将折扣奖励列表转换为 NumPy 数组

                advantage = critic.get_advantage(bs, br)  # 计算优势函数
                advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-6)  # 标准化优势函数
                for k in range(K_epoch):  # 对每个回合进行 K_epoch 次训练
                    actor.learn(bs, ba, advantage)  # 更新 Actor 网络
                    critic.learn(bs, br)  # 更新 Critic 网络
                actor.update_policy()  # 更新策略网络
                print("episode: {}/{}, score: {}".format(e + 1, episodes, score))  # 打印当前回合的得分
                break  # 跳出回合循环
        plot_score.append(score)  # 将当前回合的得分添加到得分列表中
    plt.plot(plot_score)  # 绘制得分曲线图
    plt.show()  # 显示图形