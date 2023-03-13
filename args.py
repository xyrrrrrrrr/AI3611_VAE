'''
This is a file to store the arguments for training and testing.

@Author: Xiangyun Rao
@Date: 2023.3.13
'''
import torch
import argparse


def get_args():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 32)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR', help='learning rate (default: 1e-3)')
    parser.add_argument('--device', type=str, default= '-1', help='device to use for training / testing')
    parser.add_argument('--save-dir', type=str, default='./model', help='directory to save model to')
    parser.add_argument('--testing', type=bool, default=False, help='whether to test the model')
    parser.add_argument('--results-dir', type=str, default='./results', help='directory to save results to')
    parser.add_argument('--model', type=str, default='./model/VAE_model.pth', help='model to use for testing')
    parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer to use for training / testing')
    args = parser.parse_args()
    return args