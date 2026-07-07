# Aprendizado baseado em instancias

> O processo de aprendizagem nesta familia de algoritmos consiste em armazenar os dados de treino

> Nos algortimos IBL a maquina de aprendizagem constroi uma aproximacao diferente da funcao objetivo para cada novo padrao de consulta

Mas como assim constroi uma aproximacao diferente da funcao objetivo para cada novo padrao ?

Para entender isso vamos relembrar o que e uma funcao objetivo?

Basicamente e uma funcao que mapeia minhas entradas X para uma saida (rotulo conhecido) Y. matematicamente dizendo temos:

$$f(x) = X \rightarrow Y $$

Por exemplo vamos supor que queremos criar um algoritmo para prever precos de imoveis basedo, entao eu criei uma funcao objetivo: $$ Price = f(area)$$

Tenho os dados por exemplo

|Area|Price|
|---|-----|
|50 m²|R$ 300 mil|
|100 m²|R$ 500 mil|
|150 m²|R$ 800 mil|

Para esse tipo de problema e muito comum utilizar um regressor linear. Entao, o que o algoritmo faz e encontrar uma funcao que melhor se ajusta aqueles dados:

$$f(x) = w_{0} + w_{1}x$$

A partir do momento que o meu algortimo consegue encontrar esta funcao o treino termina e nao se gera mais novas funcoes objetivo. Na realidade o que ira ocorrer e quando chegar uma novo dado de area por exemplo: **area = 300 m²**
ele simplesmente executara: $f(300)$ e retornara o resultado a funcao esta pronta ela nao muda.

> Entao neste tipo de problema o que de fato esta ocorrendo e o aprendizado de uma funcao global (aprendizado global)

**Portanto nos algoritmos utilizando a ideia de Istance based learning, temos um aprendizado local.**

**Entao sempre que chegar um novo exemplo (classe) de um conjunto de dados ele vai me gerar uma funcao de objetivo diferente que e basicamente uma aproximacao baseada em uma medida de distancia**

Entao, isso e o que difere estes algoritmos das demais tecnicas dentro do contexo de maquinas de aprendizagem supervisionadas, como por exemplo as Redes Neurais, Arvores de Decisao etc

## k-Nearest Neighbor

O kNN, e um dos algoritmos mais fomosos dentro da familia de algoritmos de aprendizagem baseado em instancias, inicialmente qual e a ideia chave ou melhor dizendo como funciona este algoritmo:

- Como visto a ideia chave dos algortimos **IBL** e apenas armazenar todos os exemplos de treinamento $$\langle x, f(x) \rangle$$

- Entao, dado um valor de consulta $x_{q}$, primeiro localize o exemplo de treinamento mais proximo $x_{i}$, e entao se estima: $$\hat{f}(x_{q}) \leftarrow f(x_{i})$$

ou seja, se estima qual valor de $x_{q}$ mais se aproxima de $x_{i}$ e como isso e possivel se determinar qual e a definicao da classe ou do valor para aquele determinado exemplo $x_{q}$

## Como e localizado o exemplo de treinamento mais proximo

Para se localizar o exemplo mais proximo e feito o seguinte:

**Dado um valor de consulta $x_{q}$, calcula-se a frequencia das classes dos $k$ vizinhos mais proximos .Portanto, o valor da consulta $x_{q}$ que esta sendo feita a "inferencia" vai pertencer a classe de maior frequencia**

Em exemplos mais explicitos suponha que escolhemos k = 5 calculamos as distancias e os 5 vizinhos mais proximos de $x_{q}$ percentecem: 

|Vizinho|Classe|
|---|----|
| 1 | A |
| 2 | B |
| 3 | A |
| 4 | A |
| 5 | B |

Logo, aqui neste caso o novo valor da consulta $x_{q}$ vai pertencer a classe A pois tem 3 ocorrencias.

> Nota: "**k**" e um parametro do algoritmo, que representa o numero de vizinhos sendo um valor arbitrario.Porem, quando estamos trabalhando com problemas de classificacao binaria e bem comum utilizar-se valores impares para reduzir a chance de empate

Estes vizinhos mais proximos sao calculados a partir de medidas de distancia onde a distancia euclidiana e a medida de distancia mais utilizada e tambem comum de se encontrar em frameworks como o sckitlearn

## Calculando a distancia dos vizinhos mais proximos

Dado um padrao x descrito pelo seguinte vetor de caracteristicas: $$ x = (x_{1},x_{2},...,x_{d})$$

- sabendo que $x_{i}$ representa o i-esimo atributo do padrao $x$ a distancia entre dois padroes $x$ e $y$ e dada por: 

$$ d(\mathbf{x}, \mathbf{y}) =
\sqrt{\sum_{i=1}^{d} (x_i - y_i)^2}$$

veja o pseudo-codigo do algoritmo k-NN:

![k-nn](../img/k-nn.png)

Esse pseudo-codigo vale para quando estamos trabalhando com problemas de classificacao, porem quando estamos trabalhando com problemas de regressao ao invez de utilizar o operador $ \argmax$ utilizamos a media veja:

$$\hat{f}(x_{q}) \leftarrow \frac{\sum^{k}_{i=1}f(x_{i})}{k}$$

## Exemplo

Veja o seguinte exemplo:

![example-knn](../img/example-knn.png)

- Simbolo "$+$" = classe positiva'
- Simbolo "$-$" = classe negativa'

Quero classificar o $x_{q}$ que e o nosso exemplo de consulta

- se **k = 1** $\rightarrow$ meu $x_{q}$ vai ser da classe positiva pois na imagem o vizinho mais frequente e proximo de $x_{q}$ e "$+$"

- se **k = 5** $\rightarrow$ meu $x_{q}$ vai ser da classe negativa pois meus 5 vizinhos mais proximos serao os que estao dentro do circulo como a maior frequencia e $"-"$ entao ele sera minha classe para k = 5

> O hiperparametro "k" que estou utilizando, vai influenciar diretamente na minha fronteira de decisao

Vamos visualizar este outro exemplo

![knn2](../img/knn2.png)

- Simbolo "+" = classe vermelha'
- Simbolo "x" = classe azul'
- Simbolo "□" = $x_{q}$

1) Primeiro passo: armazeno todos os meus dados de treinamento

2) Segundo passo: calcular as distancias de $x_{q}$ entre todos os exemplos do meu conjunto de treinamento ("cruz e x");

3) Teceiro passo: escolha dos k vizinhos:

- se k = 1: o vizinho mais proximo e um "x" entao o meu $x_{q}$ e pertecente a classe azul

![k1](../img/k1.png)

- se k = 3: os vizinho mais proximos sao uma "+" entao o $x_{q}$ pertence a classe vermelha

![k2](../img/k3.png)

- se k = 9: os vizinho mais proximos sao "x" entao o $x_{q}$ pertence a classe azul

![k9](../img/k9.png)


> A escolha da distancia ou seja, se utiliza-se distancia euclidiancia, distancia de manhatann ou etc vai depender da dimensionalidade dos dados da escala e ate mesmo do formato da vizinhanca. Portanto, percebe-se que o kNN e um algoritmo sensivel a escala dos dados