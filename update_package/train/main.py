import argparse
import os
import time

import torch

from apis import generate_dataset, STGCN, mape_loss, load_data_15min, split_dataset, generate_dataset_speed_smooth
from utils import progress_bar


def prepare(args):
    """
    数据与模型准备
    """
    file_name = args['data_path'] + args['train_name'] + '.csv'
    raw_data = load_data_15min(file_name)

    # 生成数据集
    if args['type'] == 'flow':
        inputs, target = generate_dataset(raw_data, num_timesteps_input=96, num_timesteps_output=96)
    else:
        inputs, target = generate_dataset_speed_smooth(raw_data , num_timesteps_input=96, num_timesteps_output=96)

    # 拆分训练数据与验证数据
    train_inputs, train_target, test_inputs, test_target = split_dataset(inputs, target)

    test_inputs = test_inputs.to(device=args['device'])
    test_target = test_target.to(device=args['device'])

    # 定义网络
    net = STGCN(num_nodes=1, num_features=train_inputs.shape[3], num_timesteps_input=96, num_timesteps_output=96)
    net = net.to(device=args['device'])

    # 定义优化器与损失函数
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    loss_criterion = mape_loss()

    return train_inputs, train_target, test_inputs, test_target, net, optimizer, loss_criterion


def train(args, epoch, train_inputs, train_target, net, criterion, optimizer):
    """
    训练
    """
    print('\nEpoch: %d' % epoch)

    batch_size = args['batch_size']
    permutation = torch.randperm(train_inputs.shape[0])
    adj_map = torch.FloatTensor([[1]]).to(device=args['device'])
    train_loss = 0
    batch_idx = 0

    for i in range(0, train_inputs.shape[0], batch_size):
        net.train()
        optimizer.zero_grad()

        indices = permutation[i:i + batch_size]
        x_batch, y_batch = train_inputs[indices], train_target[indices]
        x_batch = x_batch.to(device=args['device'])
        y_batch = y_batch.to(device=args['device'])

        out = net(adj_map, x_batch)
        # 计算损失函数
        loss = criterion(out, y_batch)
        # 反向传播与更新权重
        loss.backward()
        optimizer.step()

        # 进度可视化
        train_loss += loss.item()
        batch_idx += 1

        if batch_idx % 10 == 0:
            progress_bar(batch_idx, int(train_inputs.shape[0] / batch_size), 'Loss: %.3f | None: %.3f%% (%d/%d)'
                         % (train_loss / (batch_idx + 1), 1, 1, 1))


def test(args, test_inputs, test_target, net, criterion, low_loss):
    test_inputs = test_inputs.to(device=args['device'])
    test_target = test_target.to(device=args['device'])
    adj_map = torch.FloatTensor([[1]]).to(device=args['device'])

    with torch.no_grad():
        net.eval()
        out = net(adj_map, test_inputs)
        test_loss = criterion(out, test_target)

    test_loss = test_loss.item()

    # 模型的保存
    if test_loss < low_loss and args['save_pth']:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'test_loss': test_loss
        }
        save_dir = args['save_path'] + args['type']
        save_name = save_dir + '/' + args['train_name']
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        torch.save(state, save_name)

    return test_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='STGCN for Traffic HLJ')
    parser.add_argument('--enable-cuda', action='store_true', help='Enable CUDA')
    parser.add_argument("--epochs", type=int, default=30, help='训练次数, 影响训练时间与精度')
    parser.add_argument("--lr", type=int, default=1e-3, help='梯度下降学习率, 影响训练精度')
    parser.add_argument("--batch_size", type=int, default=64, help='并行训练数据量, 影响训练时间与精度, 不宜过小, 依赖于内存/显存大小')
    parser.add_argument("--data_path", type=str, default='output_data/', help='数据存放路径')
    parser.add_argument("--save_path", type=str, default='checkpoints/', help='模型保存路径')
    parser.add_argument("--train_name", type=str, default='G1-1_up', help='需要训练的路段名字')
    parser.add_argument("--type", type=str, default='speed', help='训练的路段类型, 流量或速度')

    parser.add_argument("--device", type=str, default='cuda', help='训练平台')
    parser.add_argument("--save_pth", type=bool, default=True, help='是否保存模型')
    args = vars(parser.parse_args())

    # 读取数据
    train_inputs, train_target, test_inputs, test_target, net, optimizer, loss_criterion = prepare(args)

    # 尝试读取已有模型精确度
    try:
        save_dir = args['save_path'] + args['type']
        save_name = save_dir + '/' + args['train_name']
        state = torch.load(save_name)
        low_loss = state['test_loss']
    except:
        low_loss = 101

    print('low loss', low_loss)
    for epoch in range(args['epochs']):
        start_time = time.time()
        train(args, epoch, train_inputs, train_target, net, loss_criterion, optimizer)
        test_loss = test(args, test_inputs, test_target, net, loss_criterion, low_loss)
        print()
        print('test loss', test_loss)

        if test_loss < low_loss:
            low_loss = test_loss

        end_time = time.time()
        print('epoch time', end_time - start_time)
