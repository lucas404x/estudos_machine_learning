from pybrain.structure.networks import FeedForwardNetwork
from pybrain.structure import FullConnection
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit

rede = FeedForwardNetwork()

camadaEntrada = LinearLayer(2) # quantidade de neuronios na camada de entrada
# Não serão submetidos a nenhuma função de ativação, por isso usar o LinearLayer


camadaOculta = SigmoidLayer(3)
camadaSaida = SigmoidLayer(1)
bias_oculta = BiasUnit()
bias_saida = BiasUnit()

rede.addModule(camadaEntrada)
rede.addModule(camadaOculta)
rede.addModule(camadaSaida)
rede.addModule(bias_oculta)
rede.addModule(bias_saida)

entradaOculta = FullConnection(camadaEntrada, camadaOculta)
ocultaSaida = FullConnection(camadaOculta, camadaSaida)
biasOculta = FullConnection(bias_oculta, camadaOculta)
biasSaida = FullConnection(bias_saida, camadaSaida)

rede.sortModules()


