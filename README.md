# cs227 - Machine Learning

Definition: 
- Arthur Samuel (1959): "É o campo de estudo que da ao computador a habilidade de aprender sem ser explicitamente programado". 

- Tom Mitchell (1997): "O aprendizado de maquina e um campo de estudo de algoritmos que melhoram ou optimizam uma performance **P** em uma tarefa **T** com uma experiencia **E**"

    - **P**: a forma que estamos utilizando para avaliar como a tarefa esta se conseguindo se adequar;
    - **T**: A tarefa que estamos tentando fazer a predicao;
    - **E**: representa os dados e/ou informacao que o modelo esta utilizando pra aprender


### Ideia Principal
A ideia principal e que a maquina(computador) tenha capacidade de aprender a partir de dados, e nao possuir regras especificas como existiam nos modelos especialista. Modelos especialista que por sua vez eram definidos por regras e comportamentos claros, nas quais nao conseguiam se adaptar a novos cenarios, ou a novos dados. 
    
Portanto, a ideia e que esses modelos de aprendizagem no fundo, "despertem" a capacidade de se adaptar a novos cenarios e nao somente conseguir aprender
atraves dos dados que foi treinado. Mas sim aprender a generalizar para todos os cenarios com novos dados que ele ainda nao observou durante o treino por exemplo.


### Aplicacoes
- Saude;
- Financas;
- Recomendacao;
- Veiculos Autonomos;
- Visao Computacional;
- Processamento de Linguagem Natural;
- Ciberseguranca

## Machine Learning Pipeline
![ml-pipeline](image.png)

Todas as etapas sao extremamente importantes, dentro do fluxo ou pipeline de um modelo de machine learning. Porem, vale destacar que as etapas `2`e `3`sao de suma importancia.

A aquisicao de dados

O Pré-processamento e muito importante porque nem sempre os nossos modelos trabalham com o mesmo tipos de dados, ou seja, eles trabalham com dados diferentes entre si, por exemplo em quando estamos trabalhando com imagens ou voz, estamos tratando de um formato de dados nao estuturado, já quando trabalhamos com dados tabulares estamos lidando com dados estruturados. 

Por exemplo, as redes neurais trabalham necessariamente com dados numericos, entao se eu tenho uma feature ou variavel categorica nos meus dados eu preciso de realizar uma serie de pre processamentos para, que tenha-se estes dados formatados da melhor maneira possivel para o modelo.

Já se pensarmos em modelos baseados em arvores, eles ja conseguem lidar com dados nominais. E se eu fizer esta fase bem feita isso pode trazer beneficios para o modelo, alem de facilitar o treinamento.

### AutoML - Automated Machine Learning
O AutoML e basicamente uma area ou um campo que ja tem embutido dentro da sua concepcao todo este ciclo de pipeline de 1 a 5, como mostrado na imagem e ele realiza isso totalmente de forma automatica, onde eles fazem a estrategia de busca de pre processamento, fazem a busca de diferentes tipos de aprendizado, fazem otimizacao dos hiperparametros. E todo esse processo e realizado de forma automatica, e uma ferramenta muito interessante para quando nao seja conhece muito qual modelo utilizar para treinar e etc.


## Exemplos basicos de aprendizado de maquina:

**"Quero construir uma maquina de aprendizado (algoritmo) que consiga prever o valor de imoveis em uma determinada regiao", uma maquina simples para resolver este problema seria o regressor linear**


![ml-regression](img/ml-regression.png)

Cada modelo trabalha de uma determinada maneira, entao mesmo que os dados sejam iguais e o problema seja o mesmo, modelos diferentes irao produzir saidas diferentes.
 

No Machine learning, nem sempre queremos ter o melhor modelos possivel, porque pode ser que quando eu passo pra este modelo um novo dado, ele pode nao conseguir generalizar para esta nova informacao. 

Logo, queremos um modelo que aprenda com os dados de treinamento, mas que tambem que seja capaz de generalizar para novos dados, que seja uma maquina de aprendizagem nao enviesada, ou seja, nao quero um modelo com alto overfitting, overfitting impacta
na capacidade de generalizacao, veja esse exemplo do mesmo problema de prever precos de imoveis:


![ml-overfit](img/ml-overfit.png)


**"Quero construir uma maquina de aprendizado que consiga classificar se o tumor e maligno vs nao-maligno" uma maquina simples seria um modelo linear tambem, porem para o problema de classificacao e nao regressao.**

Veja que os modelos de regressao linear, nao sao utilizados apenas para problemas de `predicao`, mas sao usados tambem para modelos de `classificacao`, ou seja ao invez de ter uma saida numerica neste caso eu tenho uma saida binaria (0 e 1)



Entao,veja a imagem abaixo:

![ml-class](img/mlc.png)

Veja que na imagem acima, temos um problema de classificacao e se utilizarmos uma maquina de aprendizado linear e fizermos um "corte" seremos capazes de separar e classificar dado os dados se o tumor e **maligno classe 1 (sim)** ou se e **nao-maligno classe 0 (nao)**.

Porém, se fizermos outra observacao na regiao onde a variavel categorica do tamanho do tumor (Tumor Size), seja de tumores com tamanho na faixa de 8 a 12, teremos dificuldade de separar as duas classes, ou seja, neste caso a maquina de aprendizado comeca a ter problemas para classificar atraves do tumor size se e maligno: classe 1 ou se e nao-maligno classe 0. Esta, dificuldade, demonstra uma certa limitacao em relacao a variavel categorica tumor size pois representa que ela somente pode nao ser suficiente para representar nosso problema, e que se talvez tivessemos mais informacoes como sintomas da doenca por exemplo, ficaria mais simples e conseguiriamos classificar, tendo um melhor modelo. Portanto, tem problemas que conseguimos resolver com maquinas de aprendizado mais simples como, regressores lineares, porem em alguns cassos sera necessario utilizar maquinas mais complexas, entao tudo depende do problema que queremos tratar para encontrar a maquina de aprendizado que queremos.

## Quando utilizar Machine Learning?

Na verdade nao existe uma regra e pode-se utilizar machine learning, em teoria em qualquer tipo de problema. Porem, o termo mais adequado seria dizer quando e recomendado se utilizar aprendizado de maquina entao:

- Existe um padrao;
- Nao existe uma funcao que resolve o problema;
- E o mais importante e crítico a existencia de **dados**;


## Componentes do aprendizado de maquina