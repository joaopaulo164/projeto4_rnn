# Metodologia Científica

> O experimento completo **não foi executado** nesta etapa. A metodologia abaixo vale para a execução futura.

## 1. Abordagem empírica

Comparações de desempenho são hipóteses empíricas, sem presunção de superioridade de modelos recorrentes, de pooling ou de capacidade.

## 2. Split treino-validação agrupado por hash

**Problema anterior.** O split por índice permitia que séries idênticas (mesmo hash) ficassem em treino e validação (24 hashes compartilhados, 275 pares; 65 com rótulos diferentes), inflando a validação e contaminando early stopping, seleção de pooling e de modelo.

**Algoritmo.**
1. Calcular o hash SHA-256 de cada uma das 8.926 séries de desenvolvimento (96 × float64 little-endian, sem rótulo/índice).
2. Agrupar linhas por hash; cada grupo é indivisível.
3. Executar `StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)` e obter cinco folds candidatos (validação = fold).
4. Pontuar cada fold: `|fração_val − 0,20| + L1(classes_val − oficiais) + L1(classes_treino − oficiais)`; folds com hash compartilhado ou classe ausente são inelegíveis; empate → menor índice.
5. Escolher o menor escore. **Nenhuma métrica de modelo participa.**
6. Verificar e persistir (`SplitIntegrityError` em qualquer violação).

Resultado: fold 2, escore 0,001228, treino 7.141, validação 1.785, 8.757 grupos, 0 hashes compartilhados.

**Rótulos conflitantes.** Grupos com o mesmo hash e rótulos oficiais diferentes permanecem inteiros em uma só partição; não se altera rótulo, não se escolhe majoritário, não se remove observação. Esses casos não podem ser separados deterministicamente usando só os 96 valores.

**Duplicatas internas.** Mantidas em treino, validação e teste (sem deduplicação e sem ponderação por grupo). Consequência: leve otimismo possível em séries repetidas e peso implícito maior para grupos grandes.

## 3. Seleção de checkpoints, pooling e modelo

- **Checkpoint/época:** menor `val_loss` (`EarlyStopping`, `patience=15`, `min_delta=0,0001`).
- **Pooling (Fase 2) e modelo final:** hierarquia: maior `val_macro_f1` médio → maior `val_balanced_accuracy` médio → menor desvio de Macro F1 → menor tempo médio de treino → menos parâmetros. **Parcimônia:** se a vantagem de Macro F1 sobre a alternativa mais simples for ≤ 0,005, escolhe-se a mais simples.
- **Seed final:** a de menor `best_val_loss` dentre as sementes do experimento escolhido (empate: menor seed); o teste oficial não é usado.

## 4. Teste oficial e sensibilidade

O teste oficial é avaliado **uma vez**, para o modelo congelado. Duas análises descritivas adicionais, nunca usadas para seleção: (a) teste sem hashes presentes no desenvolvimento oficial; (b) teste sem grupos com rótulos conflitantes. Detalhes em [`RESULTS_GUIDE.md`](RESULTS_GUIDE.md).

## 5. Tempos e proveniência

Tempos obrigatórios (total, por época, inferência) são consolidados com média, desvio, mínimo e máximo; ausência ou tipo inválido interrompe a consolidação (nunca vira zero). A seleção final e o teste registram experimento, seed, checkpoint, SHA-256, melhor época, hashes de configuração/dataset/split e timestamp.

## 6. Sementes

`TRAIN_SEEDS = [7, 21, 42, 84, 168]` medem a estabilidade da otimização sobre uma partição fixa; média e desvio não são intervalos de confiança populacionais.

## 7. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Comandos para **execução posterior pelo usuário** (não executados aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
