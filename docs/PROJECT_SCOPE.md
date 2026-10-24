# Escopo do Projeto - Projeto 4 (ElectricDevices RNN)

## 1. Visão Geral

O **Projeto 4** é uma implementação acadêmica e científica rigorosa destinada à **classificação multiclasse de séries temporais de consumo elétrico (ElectricDevices - 7 classes)** utilizando Redes Neurais Recorrentes (RNN - GRU e LSTM) e baselines neurais (MLP), desenvolvida em **Python 3.11+ e PyTorch**.

## 2. Perguntas de Pesquisa (Research Questions)

O estudo é estruturado para produzir evidências empíricas reprodutíveis e responder a seis questões de pesquisa:

- **RQ1. A recorrência supera a MLP?**
  Compara o baseline MLP-32 (96 -> 32 -> 7) contra o melhor modelo GRU-32.
- **RQ2. Qual estratégia de pooling funciona melhor na GRU?**
  Ablação de 4 estratégias temporais: *último estado*, *mean pooling*, *max pooling* e *mean + max pooling*.
- **RQ3. GRU e LSTM diferem quando ambas possuem 32 unidades ocultas?**
  Comparação entre células recorrentes fixando a dimensão oculta (32) e a melhor estratégia de pooling.
- **RQ4. Aumentar a capacidade de 32 para 64 unidades traz ganho?**
  Avaliação de escalabilidade e ganho de capacidade para GRU e LSTM.
- **RQ5. A diferença entre GRU e LSTM permanece quando a capacidade aumenta?**
  Comparação das duas células recorrentes na capacidade expandida (64 unidades).
- **RQ6. O possível ganho justifica o aumento do número de parâmetros e do tempo computacional?**
  Análise conjunta de Macro F1, Balanced Accuracy, número de parâmetros, tempo por época, tempo de inferência e tempo total com aplicação do critério de parcimônia (margem de 0.005 Macro F1).

## 3. Entregáveis do Projeto

1. **Código-fonte modular e tipado** em `src/project4_rnn/`.
2. **Bateria experimental de 40 treinamentos** (8 configurações x 5 sementes).
3. **Orquestrador automatizado com reentrância e grafo de dependências**.
4. **Catálogo de evidências científicas, tabelas e figuras em alta resolução**.
5. **Relatório automatizado de auditoria e conformidade**.
