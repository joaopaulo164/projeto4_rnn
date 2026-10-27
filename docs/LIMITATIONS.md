# Limitações

> O experimento completo **não foi executado** nesta etapa; as limitações abaixo descrevem o desenho e os dados, não resultados.

## 1. Duplicatas e rótulos conflitantes

- O ElectricDevices oficial contém séries idênticas: 16.160 únicas em 16.637 amostras (477 excessos, 136 grupos duplicados).
- 54 grupos (366 amostras) têm **rótulos conflitantes**: os mesmos 96 valores aparecem com classes diferentes. Isso não pode ser resolvido deterministicamente apenas com os valores; os rótulos oficiais não foram alterados, nenhum rótulo majoritário foi escolhido e nenhuma observação foi removida.
- No split novo, cada hash fica integralmente em treino **ou** validação (0 hashes compartilhados). Duplicatas **internas** a cada partição foram mantidas (sem deduplicação e sem ponderação por grupo), logo uma partição pode conter séries repetidas e as métricas internas podem ser levemente otimistas ou pessimistas conforme o conflito.

## 2. Sobreposição com o teste oficial

Desenvolvimento oficial × teste oficial compartilham 57 hashes (178 amostras de desenvolvimento, 208 de teste, 41 grupos com conflito). O teste não foi modificado; as sensibilidades futuras (sem sobreposição exata, sem conflitos) são apenas descritivas.

## 3. Split

- O split é um único fold (de 5 candidatos) escolhido por critério de tamanho/classes/integridade; as proporções por classe não são exatamente iguais às oficiais.
- Tamanhos observados: 7.141 treino e 1.785 validação (19,998%), mas o protocolo não presume esses números.
- Resultados da versão anterior (split por índice, com 24 hashes compartilhados) são incompatíveis e foram invalidados pelo `split_hash`.

## 4. Estatísticas

- Cinco sementes avaliam estabilidade de otimização, não variabilidade amostral.
- Um único split de validação e um único teste.
- Regra de parcimônia de 0,5 p.p. de F1 macro é uma convenção do projeto.

## 5. Escopo

Uma base, um domínio (consumo elétrico), sete classes desbalanceadas, CPU; arquiteturas pequenas (MLP-32, GRU/LSTM 32 e 64). Não se generaliza para outras bases.

## 6. Desenvolvimento

Validações desta etapa usam fixtures sintéticas; a cobertura de testes foi medida em 84%, mas isso não substitui a execução experimental completa.

## 7. PRÓXIMA EXECUÇÃO EXPERIMENTAL

Para execução posterior pelo usuário (**não executados** aqui):

```powershell
python -m project4_rnn.cli clean-artifacts --generated-only

python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --fail-fast

# em caso de interrupção
python -m project4_rnn.cli run-full-experiment --config configs/base.yaml --resume --fail-fast
```
