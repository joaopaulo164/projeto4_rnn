# Limitações Científicas e Metodológicas Obrigatórias

Para assegurar honestidade intelectual e rigor científico, documentam-se as seguintes limitações do estudo:

1. **Benchmark de Classificação de Séries Temporais**: Os resultados referem-se ao benchmark oficial ElectricDevices do UCR/UEA Archive (16.637 séries temporais, 7 classes, $L=96$).
2. **Divisão de Desenvolvimento e Resolução Estatística**: A partição de validação (~1.785 amostras, 20% das 8.926 de treino oficial) fornece base robusta, mas variações inferiores à margem de parcimônia de 0.005 Macro F1 são tratadas como equivalência prática.
3. **Divisão de Desenvolvimento com Semente Fixa**: Foi utilizada uma partição estratificada de treino/validação com `SPLIT_SEED = 42`.
4. **Interpretação das Sementes**: As 5 sementes (7, 21, 42, 84, 168) avaliam a estabilidade da convergência do otimizador e inicialização de pesos sobre a mesma partição fixa, e **não** constituem intervalos de confiança populacionais ou validação cruzada k-fold.
5. **Premissa de Transferência de Pooling**: O pooling vencedor selecionado na Fase 2 com base na arquitetura GRU-32 é aplicado às fases dependentes (LSTM e 64 unidades). É teoricamente possível que a estratégia ótima para GRU apresente comportamento distinto em LSTM em diferentes capacidades.
6. **Diferença Estrutural de Parâmetros**: A célula LSTM possui 4 portas contra 3 portas da GRU, de modo que modelos com a mesma dimensão oculta não possuem contagem idêntica de parâmetros.
