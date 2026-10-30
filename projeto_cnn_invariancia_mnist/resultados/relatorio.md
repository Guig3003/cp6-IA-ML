# Projeto: CNNs sao realmente invariantes?

## 1. Introducao
Este experimento testa se CNNs treinadas do zero no MNIST mantem o desempenho quando a imagem sofre transformacoes visuais.

## 2. Metodologia
Foram comparadas as arquiteturas `simple` e `deep`, cada uma treinada sem augmentation (`plain`) e com augmentation (`augmented`) por 8 epocas, no dispositivo `cpu`. A validacao usa 5.000 imagens fixadas por semente. Os testes sao original, deslocado, rotacionado, com ruido, contraste alterado e combinado.

## 3. Resultados
| modelo | treino | teste | loss | acuracia |
|---|---|---|---:|---:|
| simple | plain | original | 0.0298 | 0.9913 |
| simple | plain | translated | 6.4100 | 0.2019 |
| simple | plain | rotated | 0.1893 | 0.9463 |
| simple | plain | noisy | 0.0592 | 0.9829 |
| simple | plain | contrast | 0.0298 | 0.9912 |
| simple | plain | combined | 2.5200 | 0.4674 |
| simple | augmented | original | 0.0434 | 0.9867 |
| simple | augmented | translated | 0.1364 | 0.9577 |
| simple | augmented | rotated | 0.1631 | 0.9453 |
| simple | augmented | noisy | 0.2865 | 0.8992 |
| simple | augmented | contrast | 0.0432 | 0.9863 |
| simple | augmented | combined | 0.1466 | 0.9536 |
| deep | plain | original | 0.0216 | 0.9941 |
| deep | plain | translated | 2.8958 | 0.4939 |
| deep | plain | rotated | 0.1600 | 0.9528 |
| deep | plain | noisy | 0.2403 | 0.9231 |
| deep | plain | contrast | 0.0213 | 0.9943 |
| deep | plain | combined | 1.2266 | 0.6623 |
| deep | augmented | original | 0.0186 | 0.9943 |
| deep | augmented | translated | 0.0360 | 0.9898 |
| deep | augmented | rotated | 0.0747 | 0.9787 |
| deep | augmented | noisy | 0.0544 | 0.9843 |
| deep | augmented | contrast | 0.0185 | 0.9942 |
| deep | augmented | combined | 0.0639 | 0.9795 |

Os arquivos `curvas_*.png`, `exemplos_*.png` e `filtros_*.png` documentam as evidencias visuais. Cada painel marca acertos em verde e erros em vermelho.

## 4. Discussao quantitativa
| modelo | treino | original | pior transformacao | pior acuracia | queda |
|---|---|---:|---|---:|---:|
| simple | plain | 0.9913 | translated (0.2019) | 0.7894 |
| simple | augmented | 0.9867 | noisy (0.8992) | 0.0875 |
| deep | plain | 0.9941 | translated (0.4939) | 0.5002 |
| deep | augmented | 0.9943 | rotated (0.9787) | 0.0156 |

A maior queda ocorreu em simple/plain com a transformacao translated: 0.7894 ponto(s) em relacao ao teste original.
O augmentation melhorou 6 de 10 comparacoes transformadas. No teste original, a variacao foi -0.0046 na CNN simples e +0.0002 na CNN profunda.
O deslocamento foi a principal fragilidade sem augmentation: a CNN simples caiu para 0.2019 e a profunda para 0.4939. Com augmentation, subiram para 0.9577 e 0.9898. A combinacao tambem foi muito prejudicial sem augmentation. Ruido foi uma excecao para a CNN simples, cuja acuracia caiu de 0.9829 para 0.8992; na CNN profunda, subiu de 0.9231 para 0.9843.
Pooling oferece tolerancia local, mas nao invariancia completa. A profundidade aumentou a acuracia original, mas so produziu robustez consistente quando combinada com augmentation. Os filtros da primeira camada sugerem detectores de bordas e tracos locais; os paineis de exemplos relacionam esses padroes aos erros.

## 5. Respostas obrigatorias
- Maior queda: deslocamento, principalmente sem augmentation.
- Augmentation: melhora fortemente deslocamento e combinacao, mas nao todos os casos da CNN simples.
- Teste original: pequena perda na CNN simples e ganho marginal na profunda.
- Profundidade: melhora o teste original, mas nao elimina a fragilidade sem augmentation.
- MaxPooling: ajuda parcialmente com deslocamentos, sem garantir invariancia.
- Os erros sao visualmente plausiveis porque transformacoes acumuladas alteram bordas, posicao e contraste.
- Invariancia e forte demais para os dados; o termo adequado e robustez parcial.

## 6. Conclusao obrigatoria
As CNNs treinadas neste projeto nao sao invariantes as transformacoes testadas. A maior queda foi de 0.7894 em relacao ao teste original, sobretudo no deslocamento sem augmentation. O augmentation reduziu essa queda em varias condicoes, mas nao tornou todos os resultados constantes. Os exemplos visuais mostram que pequenas mudancas de posicao, rotacao, ruido e sua combinacao podem produzir erros mesmo quando o digito continua reconhecivel. Portanto, a conclusao tecnica e de robustez parcial, dependente da arquitetura e das transformacoes vistas no treinamento.