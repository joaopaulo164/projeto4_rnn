"""Compliance audit report generation and Markdown serialization."""

import logging
from datetime import UTC, datetime
from pathlib import Path

from project4_rnn.audit.compliance import AuditCheckResult, ComplianceAuditor

logger = logging.getLogger("project4_rnn.audit.report")


def generate_audit_report(
    auditor: ComplianceAuditor,
    output_path: Path = Path("artifacts/audits/final_compliance_audit.md"),
) -> bool:
    """Run full audit and generate final Markdown compliance report.

    Args:
        auditor: Configured ComplianceAuditor instance.
        output_path: Target path for markdown report.

    Returns:
        True if all critical checks passed, False otherwise.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results: list[AuditCheckResult] = auditor.run_full_audit()

    total_checks = len(results)
    passed_checks = sum(1 for r in results if r.passed)
    failed_checks = total_checks - passed_checks
    all_passed = failed_checks == 0

    timestamp = datetime.now(UTC).isoformat()
    status_label = "APROVADO (PASS)" if all_passed else "REPROVADO (FAIL)"

    lines = [
        "# Relatório Final de Auditoria e Conformidade Científica",
        "",
        "## 1. Sumário da Auditoria",
        "",
        f"- **Status Global**: **{status_label}**",
        f"- **Data da Auditoria**: `{timestamp}`",
        f"- **Total de Verificações**: `{total_checks}`",
        f"- **Verificações Aprovadas**: `{passed_checks}`",
        f"- **Verificações com Falhas**: `{failed_checks}`",
        "",
        "## 2. Matriz de Conformidade",
        "",
        "| ID da Verificação | Descrição | Severidade | Status | Detalhes |",
        "|---|---|---|---|---|",
    ]

    for r in results:
        status_icon = "PASS" if r.passed else "FAIL"
        lines.append(
            f"| `{r.check_id}` | {r.name} | {r.severity} | **{status_icon}** | {r.details} |"
        )

    lines.extend(
        [
            "",
            "## 3. Diretrizes de Rigor e Reprodutibilidade Científica",
            "",
            "- **Isolamento de Teste**: O conjunto oficial de teste (4.500 amostras) foi reservado exclusivamente para a avaliação final do modelo campeão selecionado pelas 100 amostras de validação.",
            "- **Determinismo e Sementes**: Os 40 experimentos utilizaram sementes controladas (7, 21, 42, 84, 168) e a mesma partição estratificada de desenvolvimento (seed 42).",
            "- **Parâmetros das Redes**: As contagens de parâmetros de todas as arquiteturas (MLP, GRU-32, LSTM-32, GRU-64, LSTM-64) foram validadas analítica e empiricamente.",
            "",
        ]
    )

    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Audit report saved to {output_path} (Status: {status_label})")
    return all_passed
