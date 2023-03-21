import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset # 텐서데이터셋
from torch.utils.data import DataLoader # 데이터로더


class Net(nn.Module):
    # # 기본변수, layer를 초기화해주는 생성자
    def __init__(self, input_dim, hidden_dim, seq_len, output_dim, layers):
        super(Net, self).__init__()
        self.hidden_dim = hidden_dim
        self.seq_len = seq_len
        self.output_dim = output_dim
        self.layers = layers

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=layers,
                            # dropout = 0.1,
                            batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=True)

        # 학습 초기화를 위한 함수

    def reset_hidden_state(self):
        self.hidden = (
            torch.zeros(self.layers, self.seq_len, self.hidden_dim),
            torch.zeros(self.layers, self.seq_len, self.hidden_dim))

    # 예측을 위한 함수
    def forward(self, x):
        x, _status = self.lstm(x)
        x = self.fc(x[:, -1])
        return x

class StockLearn:
    def __init__(self, data_dim=5, hidden_dim=10, output_dim=1, learning_rate=0.01, nb_epochs=100,
                 seq_length=7, device='cpu'):
        # 설정값
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.learning_rate = learning_rate
        self.nb_epochs = nb_epochs
        self.device = device
        self.seq_length = seq_length


        self.net = Net(5, hidden_dim=hidden_dim, seq_len=self.seq_length, output_dim=output_dim, \
                       layers=1).to(self.device)
        self.datasplit()     ## load training data
        self.dataScaling()
        self.np_to_Tensor()  ## get training data

        self.model, self.train_hist = self.train_model()


    def datasplit(self):

        # 데이터 불러오기
        self.df = pd.read_csv('./data-02-stock_daily.csv')

        # 7일간의 데이터가 입력으로 들어가고 batch size는 임의로 지정
        self.batch = 100

        # 데이터를 역순으로 정렬하여 전체 데이터의 70% 학습, 30% 테스트에 사용
        df = self.df[::-1]
        train_size = int(len(df)*0.7)
        self.train_set = df[0:train_size]
        self.test_set = df[train_size-self.seq_length:]

        print(self.train_set.head())
        print(self.test_set.head())

        return

    def dataScaling(self):
        train_set = self.train_set
        test_set = self.test_set
        # Input scale
        scaler_x = MinMaxScaler()
        scaler_x.fit(train_set.iloc[:, :-1])

        train_set.iloc[:, :-1] = scaler_x.transform(train_set.iloc[:, :-1])
        test_set.iloc[:, :-1] = scaler_x.transform(test_set.iloc[:, :-1])

        # Output scale
        self.scaler_y = MinMaxScaler()
        self.scaler_y.fit(train_set.iloc[:, [-1]])

        self.train_set.iloc[:, -1] = self.scaler_y.transform(train_set.iloc[:, [-1]])
        self.test_set.iloc[:, -1] = self.scaler_y.transform(test_set.iloc[:, [-1]])

        return

    # 데이터셋 생성 함수
    def build_dataset(self, time_series, seq_length):
        dataX = []
        dataY = []
        for i in range(0, len(time_series) - seq_length):
            _x = time_series[i:i + seq_length, :]
            _y = time_series[i + seq_length, [-1]]
            # print(_x, "-->",_y)
            dataX.append(_x)
            dataY.append(_y)

        return np.array(dataX), np.array(dataY)

    def np_to_Tensor(self):
        trainX, trainY = self.build_dataset(np.array(self.train_set), self.seq_length)
        testX, testY = self.build_dataset(np.array(self.test_set), self.seq_length)

        # 텐서로 변환
        trainX_tensor = torch.FloatTensor(trainX)
        trainY_tensor = torch.FloatTensor(trainY)

        self.testX_tensor = torch.FloatTensor(testX)
        self.testY_tensor = torch.FloatTensor(testY)

        # 텐서 형태로 데이터 정의
        self.dataset = TensorDataset(trainX_tensor, trainY_tensor)

        # 데이터로더는 기본적으로 2개의 인자를 입력받으며 배치크기는 통상적으로 2의 배수를 사용
        self.dataloader = DataLoader(self.dataset,
                                batch_size=self.batch,
                                shuffle=True,
                                drop_last=True)

        return True

    def train_model(self, verbose=10, patience=10):

        criterion = nn.MSELoss().to(self.device)
        optimizer = optim.Adam(self.net.parameters(), lr=self.learning_rate)


        # epoch마다 loss 저장
        nb_epochs = self.nb_epochs
        train_hist = np.zeros(nb_epochs)
        train_data = self.dataloader


        for epoch in range(nb_epochs):
            avg_cost = 0
            total_batch = len(train_data)

            for batch_idx, samples in enumerate(train_data):
                x_train, y_train = samples

                # seq별 hidden state reset
                self.net.reset_hidden_state()

                # H(x) 계산
                outputs = self.net(x_train)

                # cost 계산
                loss = criterion(outputs, y_train)

                # cost로 H(x) 개선
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                avg_cost += loss / total_batch

            train_hist[epoch] = avg_cost

            if epoch % verbose == 0:
                print('Epoch:', '%04d' % (epoch), 'train loss :', '{:.4f}'.format(avg_cost))

            # patience번째 마다 early stopping 여부 확인
            if (epoch % patience == 0) & (epoch != 0):

                # loss가 커졌다면 early stop
                if train_hist[epoch - patience] < train_hist[epoch]:
                    print('\n Early Stopping')

                    break

        return self.net.eval(), train_hist


    def train(self):
        # 모델 학습
        # 설정값
        data_dim = 5
        hidden_dim = 10
        output_dim = 1
        learning_rate = 0.01
        nb_epochs = 100


    def test(self):
        # 예측 테스트
        with torch.no_grad():
            pred = []
            for pr in range(len(self.testX_tensor)):
                self.net.reset_hidden_state()

                predicted = self.net(torch.unsqueeze(self.testX_tensor[pr], 0))
                predicted = torch.flatten(predicted).item()
                pred.append(predicted)

            # INVERSE
            pred_inverse = self.scaler_y.inverse_transform(np.array(pred).reshape(-1, 1))
            testY_inverse = self.scaler_y.inverse_transform(self.testY_tensor)

            fig = plt.figure(figsize=(8, 3))
            plt.plot(np.arange(len(pred_inverse)), pred_inverse, label='pred')
            plt.plot(np.arange(len(testY_inverse)), testY_inverse, label='true')
            plt.title("Loss plot")
            plt.show()

    def model_save(self, model_file_name="./Timeseries_LSTM_data-02-stock_daily_.pth"):
        # 모델 저장
        torch.save(self.net.state_dict(), model_file_name)

    def model_load(self, model_name):
        # 불러오기
        model = Net(self.data_dim, self.hidden_dim, self.seq_length, self.output_dim, 1).to(self.device)
        model.load_state_dict(torch.load(model_name), strict=False)
        model.eval()

    def show_hist(self):
        # epoch별 손실값
        fig = plt.figure(figsize=(10, 4))
        plt.plot(self.train_hist, label="Training loss")
        plt.legend()
        plt.show()







def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press F9 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    ST_training = StockLearn()

    ST_training.train()

    ST_training.show_hist()

    ST_training.test()


