# Projeto CNNs: invariância no MNIST

Projeto experimental em PyTorch para investigar se redes neurais convolucionais são realmente invariantes a deslocamentos, rotações, ruído, mudanças de contraste e combinações dessas transformações.

## O que foi implementado

- duas CNNs treinadas do zero: `SimpleCNN` e `DeepCNN`;
- treinamento sem augmentation e com augmentation;
- avaliação nos seis conjuntos de teste exigidos;
- curvas de loss e acurácia de treino e validação;
- tabela CSV com os resultados dos 24 experimentos;
- exemplos visuais de acertos e erros;
- filtros da primeira camada convolucional;
- relatório final com análise quantitativa e conclusão conceitual.

## Como executar

Use Python 3.10 ou superior. Na raiz do projeto, instale as dependências:

```bash
python -m pip install torch torchvision matplotlib numpy
```

Execute o script completo:

```bash
python projeto_cnn_invariancia_mnist/projeto_cnn_completo.py \
  --epochs 8 \
  --batch-size 128 \
  --workers 0 \
  --output-dir projeto_cnn_invariancia_mnist/resultados
```

O MNIST será baixado automaticamente na primeira execução. Para testar rapidamente o pipeline, use `--epochs 1`.

O script de apoio pode ser executado assim:

```bash
python projeto_cnn_invariancia_mnist/mnist_cnn_invariancia_augmentation.py
```

## Resultados já obtidos

Os resultados da execução em CPU estão na pasta [`projeto_cnn_invariancia_mnist/resultados`](./projeto_cnn_invariancia_mnist/resultados/), incluindo a tabela [`resultados.csv`](./projeto_cnn_invariancia_mnist/resultados/resultados.csv) e o [`relatorio.md`](./projeto_cnn_invariancia_mnist/resultados/relatorio.md).

A conclusão experimental é que as CNNs apresentam robustez parcial, e não invariância completa. Sem augmentation, o deslocamento causou grandes quedas de acurácia; com augmentation, a robustez melhorou de forma significativa, especialmente para deslocamentos e transformações combinadas.
