#!/usr/bin/env python3
"""Executa clusterização geral e clusterização exclusiva de clientes ativos.

Gera somente quatro CSVs: clientes e resumo para cada uma das duas análises.
"""

from __future__ import annotations

import argparse
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
    from sklearn.cluster import KMeans
    from sklearn.impute import SimpleImputer
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import StandardScaler
except ImportError as error:
    raise SystemExit("Dependências ausentes. Execute: python3 -m pip install -r requirements.txt") from error


COLUNA_ID = "id_cliente"
COLUNA_RECENCIA = "recencia_dias"
LIMITE_ATIVO_DIAS = 180
SEMENTE_ALEATORIA = 42
N_INIT = 20
AMOSTRA_SILHOUETTE = 5000

# id_cliente e valor_total_liquido_12m não entram no algoritmo.
FEATURES_CLUSTER = [
    "recencia_dias", "n_pedidos_12m", "ticket_medio", "intervalo_medio_pedidos",
    "variacao_frequencia_90d", "proporcao_pedidos_cupom", "desconto_medio_pct",
    "pct_pedidos_atrasados", "atraso_medio_dias", "taxa_abertura_email",
    "taxa_cliques", "sessoes_site_app", "carrinhos_abandonados", "taxa_tickets_sac",
    "taxa_tickets_nao_resolvidos", "share_presentes_kits", "n_categorias_compradas",
]
COLUNAS_MEDIANA = ["recencia_dias", "n_pedidos_12m", "ticket_medio", "valor_total_liquido_12m"]


def ler_dados(caminho: Path) -> pd.DataFrame:
    """Lê a base e converte as métricas usadas pelo K-Means para números."""
    if not caminho.is_file():
        raise SystemExit(f"Arquivo não encontrado: {caminho}")
    dados = pd.read_csv(caminho)
    obrigatorias = [COLUNA_ID, *FEATURES_CLUSTER, "valor_total_liquido_12m"]
    ausentes = [coluna for coluna in obrigatorias if coluna not in dados.columns]
    if ausentes:
        raise SystemExit("Colunas obrigatórias ausentes: " + ", ".join(ausentes))
    if dados.empty or dados[COLUNA_ID].isna().any() or dados[COLUNA_ID].duplicated().any():
        raise SystemExit(f"'{COLUNA_ID}' deve conter identificadores únicos e não vazios.")

    for coluna in dict.fromkeys([*FEATURES_CLUSTER, "valor_total_liquido_12m"]):
        dados[coluna] = pd.to_numeric(dados[coluna], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return dados


def preparar_matriz(dados: pd.DataFrame) -> np.ndarray:
    """Imputa ausências pela mediana e padroniza as features antes do K-Means."""
    imputada = SimpleImputer(strategy="median").fit_transform(dados[FEATURES_CLUSTER])
    return StandardScaler().fit_transform(imputada)


def escolher_k(matriz: np.ndarray, k_min: int, k_max: int, k_fixo: int | None) -> int:
    """Escolhe K entre 2 e 8, conciliando silhouette, tamanho útil e simplicidade."""
    limite = min(k_max, len(matriz) - 1)
    if k_min < 2 or limite < k_min:
        raise SystemExit("Não há clientes suficientes para testar os valores de K solicitados.")
    avaliacoes: list[dict[str, float | int]] = []
    for k in range(k_min, limite + 1):
        modelo = KMeans(n_clusters=k, random_state=SEMENTE_ALEATORIA, n_init=N_INIT)
        labels = modelo.fit_predict(matriz)
        avaliacoes.append({
            "k": k,
            "silhouette": float(silhouette_score(
                matriz, labels,
                sample_size=min(AMOSTRA_SILHOUETTE, len(matriz)),
                random_state=SEMENTE_ALEATORIA,
            )),
            "menor_grupo": int(pd.Series(labels).value_counts().min()),
        })
    tabela = pd.DataFrame(avaliacoes)
    if k_fixo is not None:
        if k_fixo not in tabela["k"].tolist():
            raise SystemExit("O K informado deve estar dentro do intervalo avaliado.")
        return k_fixo

    minimo_util = max(30, int(np.ceil(len(matriz) * 0.02)))
    validos = tabela[tabela["menor_grupo"] >= minimo_util]
    validos = validos if not validos.empty else tabela
    melhor_silhouette = validos["silhouette"].max()
    return int(validos[validos["silhouette"] >= melhor_silhouette * 0.95].sort_values("k").iloc[0]["k"])


def clusterizar(dados: pd.DataFrame, k_min: int, k_max: int, k_fixo: int | None) -> tuple[pd.DataFrame, int]:
    """Aplica K-Means à base recebida e devolve uma cópia com a coluna cluster."""
    if len(dados) < 3:
        raise SystemExit("São necessários ao menos 3 clientes para clusterizar.")
    matriz = preparar_matriz(dados)
    k = escolher_k(matriz, k_min, k_max, k_fixo)
    resultado = dados.copy()
    resultado["cluster"] = KMeans(n_clusters=k, random_state=SEMENTE_ALEATORIA, n_init=N_INIT).fit_predict(matriz)
    return resultado, k


def resumir(dados: pd.DataFrame) -> pd.DataFrame:
    """Calcula quantidade, médias das features e medianas de métricas essenciais."""
    medias = dados.groupby("cluster")[FEATURES_CLUSTER].mean()
    medianas = dados.groupby("cluster")[COLUNAS_MEDIANA].median().add_prefix("mediana_")
    resumo = medias.join(medianas).reset_index()
    resumo.insert(1, "quantidade_clientes", dados.groupby("cluster").size().values)
    return resumo


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrada", type=Path, default=Path("data/clientes_cluster.csv"))
    parser.add_argument("--pasta-saida", type=Path, default=Path("resultados"))
    parser.add_argument("--clusters", type=int, help="K fixo para ambas as análises")
    parser.add_argument("--clusters-geral", type=int, help="K fixo apenas para a análise geral")
    parser.add_argument("--clusters-ativos", type=int, help="K fixo apenas para os ativos")
    parser.add_argument("--k-min", type=int, default=2)
    parser.add_argument("--k-max", type=int, default=8)
    args = parser.parse_args()

    dados = ler_dados(args.entrada)
    dados["status"] = np.where(dados[COLUNA_RECENCIA] <= LIMITE_ATIVO_DIAS, "ativo", "inativo")
    ativos = dados[dados["status"] == "ativo"].copy()
    k_geral_fixo = args.clusters_geral if args.clusters_geral is not None else args.clusters
    k_ativos_fixo = args.clusters_ativos if args.clusters_ativos is not None else args.clusters

    geral, k_geral = clusterizar(dados, args.k_min, args.k_max, k_geral_fixo)
    apenas_ativos, k_ativos = clusterizar(ativos, args.k_min, args.k_max, k_ativos_fixo)

    args.pasta_saida.mkdir(parents=True, exist_ok=True)
    geral.to_csv(args.pasta_saida / "clientes_clusterizados_geral.csv", index=False)
    resumir(geral).to_csv(args.pasta_saida / "resumo_clusters_geral.csv", index=False)
    apenas_ativos.to_csv(args.pasta_saida / "clientes_clusterizados_ativos.csv", index=False)
    resumir(apenas_ativos).to_csv(args.pasta_saida / "resumo_clusters_ativos.csv", index=False)

    print(f"Clusterização geral: {len(geral)} clientes, K={k_geral}.")
    print(f"Clusterização de ativos: {len(apenas_ativos)} clientes, K={k_ativos}.")
    print(f"Quatro arquivos salvos em: {args.pasta_saida}")


if __name__ == "__main__":
    main()
