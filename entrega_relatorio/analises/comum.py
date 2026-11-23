"""
Funções compartilhadas pelas análises (ACH0042 - Grupo 11).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

RAIZ = Path(__file__).resolve().parent
DADOS = RAIZ / "dados"          # painel_municipio_ano.csv, base_limpa.csv, serie_mensal.csv, municipios_coord.csv
SAIDA = RAIZ / "resultados"
SAIDA.mkdir(exist_ok=True)
K_VIZINHOS = 10


def ler_painel():
    p = pd.read_csv(DADOS / "painel_municipio_ano.csv", sep=";", encoding="utf-8-sig",
                    dtype={"cod_mun_ibge7": str, "cod_mun_ibge6": str})
    return p.sort_values(["cod_mun_ibge7", "ano"]).reset_index(drop=True)


def vizinhos(codigos, k=K_VIZINHOS):
    """k vizinhos mais próximos pelo centróide (distância na esfera). Devolve matriz de índices (n, k)."""
    c = pd.read_csv(DADOS / "municipios_coord.csv", dtype={"codigo_ibge": str}).set_index("codigo_ibge")
    c = c.loc[codigos]
    lat, lon = np.radians(c["latitude"].values), np.radians(c["longitude"].values)
    xyz = np.c_[np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
    _, idx = cKDTree(xyz).query(xyz, k=k + 1)
    return idx[:, 1:], c[["latitude", "longitude"]].reset_index(drop=True)


def esperado_populacional(p, col="registros_lgbt"):
    """E_it = total nacional do ano x participação do município na população do ano (cenário nacional)."""
    tot = p.groupby("ano")[col].transform("sum")
    pop = p.groupby("ano")["populacao"].transform("sum")
    return tot * p["populacao"] / pop


def bayes_empirico(y, e):
    """Suavização Poisson-Gama (estimador de momentos de Marshall, 1991).
    Devolve RR suavizado, P(RR > 1) e os parâmetros (alfa, beta) da priori."""
    y, e = np.asarray(y, float), np.asarray(e, float)
    r = y / e
    m = y.sum() / e.sum()
    s2 = np.sum(e * (r - m) ** 2) / e.sum()
    var = max(s2 - m / e.mean(), 1e-6)
    beta = m / var
    alfa = m * beta
    rr = (y + alfa) / (e + beta)
    p_acima = 1 - stats.gamma.cdf(1.0, a=y + alfa, scale=1 / (e + beta))
    return rr, p_acima, alfa, beta


def desvio_poisson(y, mu):
    y, mu = np.asarray(y, float), np.clip(np.asarray(mu, float), 1e-9, None)
    t = np.where(y > 0, y * np.log(y / mu), 0.0)
    return 2 * np.mean(t - (y - mu))


def topk_recall(y, mu, k=50):
    """Das k cidades com mais registros observados, quantas estão entre as k previstas."""
    obs = set(np.argsort(-np.asarray(y))[:k])
    prev = set(np.argsort(-np.asarray(mu))[:k])
    return len(obs & prev) / k


def nb_sf(y, mu, theta):
    """P(Y >= y) sob Binomial Negativa com média mu e tamanho theta."""
    p = theta / (theta + mu)
    return stats.nbinom.sf(np.asarray(y) - 1, theta, p)


def nb_quantil(q, mu, theta):
    p = theta / (theta + mu)
    return stats.nbinom.ppf(q, theta, p)
