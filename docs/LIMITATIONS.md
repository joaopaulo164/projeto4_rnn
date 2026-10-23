# Limitações Científicas e Metodológicas Obrigatórias

Para assegurar honestidade intelectual e rigor científico, documentam-se as seguintes limitações do estudo:

1. **Adaptação para Tarefa Binária**: A transformação das 5 classes originais em uma tarefa binária (Classe 1 vs Classes 2-5) modifica o problema original. Os resultados deste estudo **não devem ser comparados diretamente com publicações que avaliam o ECG5000 na tarefa multiclasse de 5 classes**.
2. **Tamanho Reduzido da Partição de Validação**: O conjunto de validação é restrito a 100 amostras (20% de 500), o que limita a resolução estatística para pequenas variações de desempenho.
3. **Divisão de Desenvolvimento Única**: Foi utilizada uma única partição estratificada de treino/validação com `SPLIT_SEED = 42`.
4. **Interpretação das Sementes**: As 5 sementes avaliam a estabilidade da convergência do otimizador e inicialização de pesos sobre a mesma partição fixa, e **não** constituem intervalos de confiança populacionais ou validação cruzada k-fold.
5. **Ausência de Validação Clínica**: As séries temporais e rótulos não foram validados clinicamente por junta médica neste estudo, destinando-se os resultados estritamente a fins acadêmicos e metodológicos de aprendizado de máquina.
6. **Premissa de Transferência de Pooling**: O pooling vencedor selecionado na Fase 2 com base na arquitetura GRU-32 é aplicado às fases dependentes (LSTM e 64 unidades). É teoricamente possível que a estratégia ótima para GRU não seja universalmente ótima para LSTM em todas as capacidades.
7. **Diferença Estrutural de Parâmetros**: A célula LSTM possui 4 portas contra 3 portas da GRU, de modo que modelos com a mesma dimensão oculta não possuem contagem idêntica de parâmetros.
