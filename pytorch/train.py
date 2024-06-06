import argparse
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from help_code_demo import ToTensor, ECG_DataSET
from models.senet import SENet
import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR



def main():
    # Hyperparameters
    BATCH_SIZE = args.batchsz
    BATCH_SIZE_TEST = args.batchsz
    LR = args.lr
    EPOCH = args.epoch
    SIZE = args.size
    path_data = args.path_data
    path_indices = args.path_indices

    # Instantiating NN
    net = SENet()
    # net = torch.load('./saved_models/current_best_0.972.pth')
    net.train()
    net = net.float().to(device)

    # Start dataset loading
    trainset = ECG_DataSET(root_dir=path_data,
                           indice_dir=path_indices,
                           mode='train',
                           size=SIZE,
                           transform=transforms.Compose([ToTensor()]))

    trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    testset = ECG_DataSET(root_dir=path_data,
                          indice_dir=path_indices,
                          mode='test',
                          size=SIZE,
                          transform=transforms.Compose([ToTensor()]))

    testloader = DataLoader(testset, batch_size=BATCH_SIZE_TEST, shuffle=True, num_workers=1)

    print("Training Dataset loading finish.")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    # scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    epoch_num = EPOCH

    Train_loss = []
    Train_acc = []
    Test_loss = []
    Test_acc = []
    max_acc = 0.0
    print("Start training")
    for epoch in range(epoch_num):  # loop over the dataset multiple times (specify the #epoch)
        net.train()
        running_loss = 0.0
        correct = 0.0
        accuracy = 0.0
        i = 0
        for j, data in enumerate(trainloader, 0):
            inputs, labels = data['ECG_seg'], data['label']
            inputs = inputs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum()
            accuracy += correct / BATCH_SIZE
            correct = 0.0

            running_loss += loss.item()
            i += 1

        print('[Epoch, Batches] is [%d, %5d] \nTrain Acc: %.5f Train loss: %.5f' %
              (epoch + 1, i, accuracy / i, running_loss / i))
        # scheduler.step()
        Train_loss.append(running_loss / i)
        Train_acc.append((accuracy / i).item())

        running_loss = 0.0
        accuracy = 0.0

        correct = 0.0
        total = 0.0
        i = 0.0
        running_loss_test = 0.0

        for data_test in testloader:
            net.eval()
            ECG_test, labels_test = data_test['ECG_seg'], data_test['label']
            ECG_test = ECG_test.float().to(device)
            labels_test = labels_test.to(device)
            outputs_test = net(ECG_test)
            _, predicted_test = torch.max(outputs_test.data, 1)
            total += labels_test.size(0)
            correct += (predicted_test == labels_test).sum()

            loss_test = criterion(outputs_test, labels_test)
            running_loss_test += loss_test.item()
            i += 1
        if correct / total > max_acc:
            max_acc = correct / total
            torch.save(net, './saved_models_senet/senet_best_{:.3f}.pth'.format(max_acc))
            torch.save(net.state_dict(), './saved_senet_sd/senet_sd_{:.3f}_light.pth'.format(max_acc))
            # torch.save(net.state_dict(), './saved_models/best_state_dict.pth')
        print('Test Acc: %.5f Test Loss: %.5f' % (correct / total, running_loss_test / i))

        Test_loss.append(running_loss_test / i)
        Test_acc.append((correct / total).item())

    # torch.save(net, './saved_models/ECG_net.pkl')
    # torch.save(net.state_dict(), './saved_models/ECG_net_state_dict.pkl')

    file = open('./saved_models/loss_acc.txt', 'w')
    file.write("Train_loss\n")
    file.write(str(Train_loss))
    file.write('\n\n')
    file.write("Train_acc\n")
    file.write(str(Train_acc))
    file.write('\n\n')
    file.write("Test_loss\n")
    file.write(str(Test_loss))
    file.write('\n\n')
    file.write("Test_acc\n")
    file.write(str(Test_acc))
    file.write('\n\n')

    print('Finish training')


if __name__ == '__main__':
        argparser = argparse.ArgumentParser()
        argparser.add_argument('--epoch', type=int, help='epoch number', default=30)
        argparser.add_argument('--lr', type=float, help='learning rate', default=0.0001)
        argparser.add_argument('--batchsz', type=int, help='total batchsz for traindb', default=32)
        argparser.add_argument('--cuda', type=int, default=0)
        argparser.add_argument('--size', type=int, default=1250)
        argparser.add_argument('--path_data', type=str, default='/root/contest/training_dataset/')
        argparser.add_argument('--path_indices', type=str, default='./data_indices/')

        args = argparser.parse_args()

        device = torch.device("cuda:" + str(args.cuda))

        print("device is --------------", device)

        main()
