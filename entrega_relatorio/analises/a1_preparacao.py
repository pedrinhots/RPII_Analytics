"""
Preparação compartilhada da Análise 1 (ACH0042 - Grupo 11): painel com valor esperado, defasagens próprias,
de vizinhos e histórico suavizado; padronização; e o modelo endêmico-epidêmico Binomial Negativo.
Usado por a1_previsao_ondas.py e a1b_aprendizado.py.
"""
import warnings

import numpy as np
import pandas as pd
from scipy import optimize, special
from statsmodels.tools.numdiff import approx_hess3

from comum import bayes_empirico, esperado_populacional, ler_painel, vizinhos

warnings.filterwarnings("ignore")
rng = np.random.default_rng(42)

# ---------------------------------------------------------------------------
# Dados e variáveis
# ---------------------------------------------------------------------------
p = ler_painel()
codigos = p["cod_mun_ibge7"].unique()
N_MUN = len(codigos)
idx_viz, coords = vizinhos(codigos)
p["e"] = esperado_populacional(p)                       # cenário nacional x participação populacional
nac = p.groupby("ano")["registros_lgbt"].sum()
p["r_nac"] = p["ano"].map(nac / nac.shift(1))           # razão do volume nacional t / t-1

Y = p.pivot(index="cod_mun_ibge7", columns="ano", values="registros_lgbt").loc[codigos]
E = p.pivot(index="cod_mun_ibge7", columns="ano", values="e").loc[codigos]
ANOS = list(Y.columns)

# defasagens: próprio município, vizinhos, histórico suavizado (só com anos < t)
lag = Y.shift(1, axis=1)
lag_viz = pd.DataFrame(Y.values[idx_viz].mean(axis=1), index=Y.index, columns=ANOS).shift(1, axis=1)
lag3 = Y.T.shift(1).rolling(3, min_periods=1).sum().T
rr_hist = pd.DataFrame(np.nan, index=Y.index, columns=ANOS)
for t in ANOS[1:]:
    anteriores = [a for a in ANOS if a < t]
    rr, _, _, _ = bayes_empirico(Y[anteriores].sum(axis=1), E[anteriores].sum(axis=1))
    rr_hist[t] = rr


def longo(df, nome):
    return df.stack().rename(nome).reset_index().rename(columns={"level_1": "ano"})


for df, nome in ((lag, "y_lag"), (lag_viz, "y_lag_viz"), (lag3, "y_lag3"), (rr_hist, "rr_hist")):
    p = p.merge(longo(df, nome), on=["cod_mun_ibge7", "ano"], how="left")

# covariáveis com defasagem onde o dado não seria conhecido no início do ano
p = p.sort_values(["cod_mun_ibge7", "ano"])
for c in ("taxa_homicidios_100mil", "pib_per_capita_r2022"):
    p[c + "_lag"] = p.groupby("cod_mun_ibge7")[c].shift(1)

p["log_pop"] = np.log(p["populacao"])
p["log_dens"] = np.log(p["densidade"])
p["log_renda"] = np.log(p["renda_pc_r2022"])
p["log_pib_lag"] = np.log(p["pib_per_capita_r2022_lag"])
p["log_hom_lag"] = np.log1p(p["taxa_homicidios_100mil_lag"])
p["log_proc_pt"] = np.log1p(p["procedimentos_pt"])
for r in ("Nordeste", "Norte", "Sul", "Centro-Oeste"):
    p["reg_" + r] = (p["regiao"] == r).astype(int)

ENDEMICAS = ["log_pop", "log_dens", "log_renda", "log_pib_lag", "log_hom_lag", "gini_2010", "pct_15_29",
             "pct_pretos_pardos", "pct_casais_mesmo_sexo_2022", "munic_indice_lgbt", "munic_ouvidoria_dh",
             "munic_estrutura_dh", "processo_transexualizador", "capital",
             "reg_Nordeste", "reg_Norte", "reg_Sul", "reg_Centro-Oeste"]
CONTINUAS = ENDEMICAS[:10]
ARVORE = ENDEMICAS + ["log_proc_pt", "l_lag", "l_lag_viz", "l_lag3", "l_rr_hist", "l_lag_esc"]

p["l_lag"] = np.log1p(p["y_lag"])
p["l_lag_viz"] = np.log1p(p["y_lag_viz"])
p["l_lag3"] = np.log1p(p["y_lag3"])
p["l_rr_hist"] = np.log(p["rr_hist"])
p["l_lag_esc"] = np.log1p(p["y_lag"] * p["r_nac"])
p = p[p["ano"] >= 2012].reset_index(drop=True)  # 2011 não tem defasagem


def padronizar(treino, teste, cols):
    mu, sd = treino[cols].mean(), treino[cols].std().replace(0, 1)
    return (treino[cols] - mu) / sd, (teste[cols] - mu) / sd


# ---------------------------------------------------------------------------
# Modelo endêmico-epidêmico Binomial Negativo (Held, Höhle & Hofmann, 2005; forma do hhh4)
#   mu_it = exp(X_it b) e_it  +  exp(a0 + a1 z_pop_i) r_t y_i,t-1  +  exp(c0 + c1 z_pop_i) r_t ybar_viz,t-1
# ---------------------------------------------------------------------------
class EndemicoEpidemico:
    def __init__(self, cols):
        self.cols = cols

    def _partes(self, theta, X, zpop, e, ar, ne):
        k = X.shape[1]
        b = theta[:k]
        a0, a1, c0, c1 = theta[k:k + 4]
        end = np.exp(np.clip(X @ b, -30, 30)) * e
        aut = np.exp(np.clip(a0 + a1 * zpop, -30, 30)) * ar
        viz = np.exp(np.clip(c0 + c1 * zpop, -30, 30)) * ne
        return end, aut, viz

    def _nll(self, theta, X, zpop, e, ar, ne, y):
        end, aut, viz = self._partes(theta, X, zpop, e, ar, ne)
        mu = end + aut + viz + 1e-10
        size = np.exp(theta[-1])
        ll = (special.gammaln(y + size) - special.gammaln(size) - special.gammaln(y + 1)
              + size * np.log(size / (size + mu)) + y * np.log(mu / (size + mu)))
        return -ll.sum()

    def _matrizes(self, d, Xz):
        X = np.c_[np.ones(len(d)), Xz.values]
        zpop = Xz["log_pop"].values
        ar = (d["y_lag"] * d["r_nac"]).values
        ne = (d["y_lag_viz"] * d["r_nac"]).values
        return X, zpop, d["e"].values, ar, ne

    def ajustar(self, d, Xz, erros=False):
        X, zpop, e, ar, ne = self._matrizes(d, Xz)
        y = d["registros_lgbt"].values
        k = X.shape[1]
        th0 = np.r_[np.zeros(k), -2.0, 0.0, -3.0, 0.0, 0.0]
        res = optimize.minimize(self._nll, th0, args=(X, zpop, e, ar, ne, y), method="L-BFGS-B",
                                options={"maxiter": 3000})
        self.theta, self.k, self.convergiu, self.nll = res.x, k, res.success, res.fun
        self.n_iter, self.n_aval = res.nit, res.nfev
        self.size = np.exp(res.x[-1])
        if erros:
            H = approx_hess3(res.x, self._nll, args=(X, zpop, e, ar, ne, y))
            self.ep = np.sqrt(np.diag(np.linalg.pinv(H)))
        return self

    def prever(self, d, Xz, partes=False):
        X, zpop, e, ar, ne = self._matrizes(d, Xz)
        end, aut, viz = self._partes(self.theta, X, zpop, e, ar, ne)
        return (end, aut, viz) if partes else end + aut + viz



LIM_ONDA_MIN, LIM_ONDA_RR = 5, 2.0  # onda = >= 5 registros e >= 2x o esperado pelo porte e cenário nacional
PARAMS_LGBM = dict(objective="tweedie", tweedie_variance_power=1.3, learning_rate=0.03, num_leaves=15,
                   min_data_in_leaf=60, feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
                   lambda_l2=1.0, verbose=-1, seed=42)
