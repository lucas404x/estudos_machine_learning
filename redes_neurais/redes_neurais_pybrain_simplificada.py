from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer

rede = buildNetwork(2, 3, 1) # 2 neuronios na camada de entrada;
                               # 3 na camada oculta;
                               # 1 na camada de saída.

dados = SupervisedDataSet(2, 1) # (quantidade de atributos, quantidade classes)
dados.addSample((0, 0), (0, ))
dados.addSample((0, 1), (1, ))
dados.addSample((1, 1), (0, ))
dados.addSample((1, 0), (1, ))

treinamento = BackpropTrainer(rede, dados, momentum = 0.06)

for i in range(30000):
    erro = treinamento.train()
    if i % 1000 == 0:
        print("Erro:", erro)

print(rede.activate([0, 0])) # predicao
print(rede.activate([0, 1])) # predicao
print(rede.activate([1, 1])) # predicao