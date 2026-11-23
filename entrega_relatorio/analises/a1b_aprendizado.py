"""
Análise 1b - Aprofundamento: como o modelo aprende e por que confiar nos resultados
ACH0042 - Grupo 11

Complementa a1_previsao_ondas.py (mesmos dados, variáveis e validação por janela móvel 2016-2022).

  1. Curvas de aprendizado do LightGBM rodada a rodada (treino x validação) e o ponto de parada antecipada
  2. Curva de tamanho do treino: desempenho conforme o número de anos usados para aprender
  3. Ablação: contribuição dos grupos de variáveis (estrutura, histórico próprio, vizinhos)
  4. Incerteza do ganho: bootstrap por município (2.000 reamostras) e consistência ano a ano
  5. Calibração (previsto x observado) e curva de captura de ondas (lift)
  6. Sensibilidade aos hiperparâmetros do LightGBM
  7. O que o modelo aprendeu: valores SHAP (amostra para os gráficos de enxame e dependência)

Saídas em resultados/a1b_*.csv e resultados/a1b_resumo.json
"""
import json
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score

from a1_preparacao import (ARVORE, ENDEMICAS, LIM_ONDA_MIN, LIM_ONDA_RR, PARAMS_LGBM, EndemicoEpidemico, p,
                           padronizar)
from comum import SAIDA, desvio_poisson, nb_sf

t_inicio = time.perf_counter()
rng = np.random.default_rng(42)
ANOS_TESTE = range(2016, 2023)
PACIENCIA = 100

GRUPOS = {
    "estrutura": ENDEMICAS + ["log_proc_pt"],
    "historico": ["l_lag", "l_lag3", "l_rr_hist", "l_lag_esc"],
    "vizinhos": ["l_lag_viz"],
}
assert set(sum(GRUPOS.values(), [])) == set(ARVORE)


def dividir(T):
    return p[p["ano"] < T].copy(), p[p["ano"] == T].copy()


def ds(df, feats, ref=None):
    return lgb.Dataset(df[feats], df["registros_lgbt"], init_score=np.log(df["e"]), reference=ref,
                       free_raw_data=False)


def ponto_de_parada(curva_val, paciencia=PACIENCIA):
    """Reproduz a regra de parada antecipada: melhor rodada até que se passem `paciencia` rodadas sem melhora."""
    melhor, i_melhor = np.inf, 0
    for i, v in enumerate(curva_val):
        if v < melhor - 1e-12:
            melhor, i_melhor = v, i
        elif i - i_melhor >= paciencia:
            break
    return i_melhor + 1  # rodadas contadas a partir de 1


def treinar_lgbm(tr, feats, params=PARAMS_LGBM):
    """Mesmo protocolo da a1: parada antecipada no último ano de treino e retreino com tudo."""
    ult = tr["ano"].max()
    trA, trB = tr[tr["ano"] < ult], tr[tr["ano"] == ult]
    if len(trA) == 0:  # um único ano de treino: sem validação interna, usa 300 rodadas
        n = 300
    else:
        dA = ds(trA, feats)
        m = lgb.train(params, dA, 3000, valid_sets=[ds(trB, feats, dA)],
                      callbacks=[lgb.early_stopping(PACIENCIA, verbose=False)])
        n = m.best_iteration
    return lgb.train(params, ds(tr, feats), n), n


def prever(m, te, feats):
    return np.exp(m.predict(te[feats], raw_score=True) + np.log(te["e"].values))


def onda_de(te):
    y = te["registros_lgbt"].values
    return ((y >= LIM_ONDA_MIN) & (y >= LIM_ONDA_RR * te["e"].values)).astype(int)


def prob_onda(te, mu, size):
    lim = np.maximum(LIM_ONDA_MIN, np.ceil(LIM_ONDA_RR * te["e"].values))
    return nb_sf(lim, mu, size)


resumo = {}

# ---------------------------------------------------------------------------
# 1. Curvas de aprendizado (1.500 rodadas registradas, sem parar, para mostrar o que vem depois)
# ---------------------------------------------------------------------------
curvas = []
for T in ANOS_TESTE:
    tr, _ = dividir(T)
    ult = tr["ano"].max()
    trA, trB = tr[tr["ano"] < ult], tr[tr["ano"] == ult]
    ev = {}
    dA = ds(trA, ARVORE)
    lgb.train({**PARAMS_LGBM, "metric": "tweedie"}, dA, 1500, valid_sets=[dA, ds(trB, ARVORE, dA)],
              valid_names=["treino", "validacao"], callbacks=[lgb.record_evaluation(ev)])
    tre, val = np.array(ev["treino"]["tweedie"]), np.array(ev["validacao"]["tweedie"])
    parada = ponto_de_parada(val)
    curvas.append(pd.DataFrame({"ano_previsto": T, "rodada": np.arange(1, len(val) + 1), "perda_treino": tre,
                                "perda_validacao": val, "rodada_escolhida": parada}))
    print(f"curva {T}: parada na rodada {parada}; validação {val[0]:.4f} -> {val[parada - 1]:.4f}; "
          f"rodada 1500: {val[-1]:.4f}")
curvas = pd.concat(curvas)
curvas.to_csv(SAIDA / "a1b_curvas_aprendizado.csv", index=False, sep=";", encoding="utf-8-sig")
r = curvas.groupby("ano_previsto").apply(lambda d: pd.Series({
    "rodada_escolhida": d["rodada_escolhida"].iat[0],
    "queda_val_ate_parada_%": (1 - d["perda_validacao"].iat[d["rodada_escolhida"].iat[0] - 1]
                               / d["perda_validacao"].iat[0]) * 100,
    "queda_treino_ate_1500_%": (1 - d["perda_treino"].iat[-1] / d["perda_treino"].iat[0]) * 100,
    "val_1500_vs_parada_%": (d["perda_validacao"].iat[-1] / d["perda_validacao"].iat[d["rodada_escolhida"].iat[0] - 1]
                             - 1) * 100}))
resumo["curvas"] = r.round(3).reset_index().to_dict("records")
print(r.round(2).to_string())

# ---------------------------------------------------------------------------
# 2. Curva de tamanho do treino (prevendo 2019 e 2022 com 1..k anos anteriores)
# ---------------------------------------------------------------------------
tam = []
for T in (2019, 2022):
    _, te = dividir(T)
    y = te["registros_lgbt"].values
    d_pop = desvio_poisson(y, te["e"].values)
    for k in range(1, T - 2012 + 1):
        tr = p[(p["ano"] >= T - k) & (p["ano"] < T)].copy()
        Xtr, Xte = padronizar(tr, te, ENDEMICAS)
        mg, n = treinar_lgbm(tr, ARVORE)
        ee = EndemicoEpidemico(ENDEMICAS).ajustar(tr, Xtr)
        for nome, mu in (("LightGBM", prever(mg, te, ARVORE)), ("Endêmico-epidêmico", ee.prever(te, Xte))):
            tam.append(dict(ano_previsto=T, anos_treino=k, linhas_treino=len(tr), modelo=nome,
                            ganho_vs_pop_pct=(1 - desvio_poisson(y, mu) / d_pop) * 100))
        print(f"tamanho {T}, {k} ano(s): LGBM {tam[-2]['ganho_vs_pop_pct']:.1f}% | EE {tam[-1]['ganho_vs_pop_pct']:.1f}%")
tam = pd.DataFrame(tam)
tam.to_csv(SAIDA / "a1b_tamanho_treino.csv", index=False, sep=";", encoding="utf-8-sig")

# ---------------------------------------------------------------------------
# 3. Ablação dos grupos de variáveis (janela móvel completa)
# ---------------------------------------------------------------------------
CONJUNTOS = {
    "Só estrutura": GRUPOS["estrutura"],
    "Só histórico da cidade": GRUPOS["historico"],
    "Estrutura + histórico": GRUPOS["estrutura"] + GRUPOS["historico"],
    "Completo (+ vizinhos)": ARVORE,
}
abl, size_por_ano = [], {}
for T in ANOS_TESTE:
    tr, te = dividir(T)
    Xtr, _ = padronizar(tr, te, ENDEMICAS)
    size_por_ano[T] = EndemicoEpidemico(ENDEMICAS).ajustar(tr, Xtr).size
    y, onda = te["registros_lgbt"].values, onda_de(te)
    d_pop = desvio_poisson(y, te["e"].values)
    for nome, feats in CONJUNTOS.items():
        mg, n = treinar_lgbm(tr, feats)
        mu = prever(mg, te, feats)
        pr = prob_onda(te, mu, size_por_ano[T])
        abl.append(dict(ano_previsto=T, conjunto=nome, n_variaveis=len(feats), arvores=n,
                        ganho_vs_pop_pct=(1 - desvio_poisson(y, mu) / d_pop) * 100,
                        ap_onda=average_precision_score(onda, pr), prec20_onda=onda[np.argsort(-pr)[:20]].mean()))
abl = pd.DataFrame(abl)
abl.to_csv(SAIDA / "a1b_ablacao_por_ano.csv", index=False, sep=";", encoding="utf-8-sig")
abl_m = abl.groupby("conjunto")[["n_variaveis", "ganho_vs_pop_pct", "ap_onda", "prec20_onda"]].mean()
abl_m = abl_m.loc[list(CONJUNTOS)]
abl_m.to_csv(SAIDA / "a1b_ablacao_resumo.csv", sep=";", encoding="utf-8-sig")
resumo["ablacao"] = abl_m.round(3).reset_index().to_dict("records")
print(abl_m.round(3).to_string())

# ---------------------------------------------------------------------------
# 4. Incerteza do ganho: bootstrap por município sobre as previsões fora da amostra da a1
# ---------------------------------------------------------------------------
pv = pd.read_csv(SAIDA / "a1_previsoes_validacao.csv", sep=";", dtype={"cod_mun_ibge7": str})
MODELOS = {"LightGBM": "mu_LightGBM (Tweedie)", "Endêmico-epidêmico": "mu_Endêmico-epidêmico (BN)",
           "Extra Trees": "mu_Extra Trees", "Só população": "mu_Só população"}


def contrib(y, mu):
    mu = np.clip(mu, 1e-9, None)
    return 2 * (np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu))


yv = pv["registros_lgbt"].values
D = {k: contrib(yv, pv[c].values) for k, c in MODELOS.items()}
mun_idx, mun = pd.factorize(pv["cod_mun_ibge7"])
soma = {k: np.bincount(mun_idx, weights=v, minlength=len(mun)) for k, v in D.items()}
B = 2000
amostras = rng.integers(0, len(mun), size=(B, len(mun)))
cont = np.stack([np.bincount(a, minlength=len(mun)) for a in amostras])  # pesos de reamostragem
tot = {k: cont @ v for k, v in soma.items()}
boot = []
for k in ("LightGBM", "Endêmico-epidêmico", "Extra Trees"):
    g = (1 - tot[k] / tot["Só população"]) * 100
    pont = (1 - D[k].sum() / D["Só população"].sum()) * 100
    boot.append(dict(comparacao=f"{k} vs só população", ganho_pct=pont, ic95_inf=np.percentile(g, 2.5),
                     ic95_sup=np.percentile(g, 97.5), p_ganho_menor_ou_igual_0=(g <= 0).mean()))
dif = (tot["Endêmico-epidêmico"] - tot["LightGBM"]) / tot["Só população"] * 100
boot.append(dict(comparacao="LightGBM vs Endêmico-epidêmico (pontos percentuais)",
                 ganho_pct=(D["Endêmico-epidêmico"].sum() - D["LightGBM"].sum()) / D["Só população"].sum() * 100,
                 ic95_inf=np.percentile(dif, 2.5), ic95_sup=np.percentile(dif, 97.5),
                 p_ganho_menor_ou_igual_0=(dif <= 0).mean()))
boot = pd.DataFrame(boot)
boot.to_csv(SAIDA / "a1b_bootstrap_ganho.csv", index=False, sep=";", encoding="utf-8-sig")
resumo["bootstrap"] = boot.round(3).to_dict("records")
print(boot.round(2).to_string())
val = pd.read_csv(SAIDA / "a1_validacao_por_ano.csv", sep=";").pivot(index="ano", columns="modelo", values="desvio")
resumo["anos_em_que_supera_populacao"] = {m: int((val[m] < val["Só população"]).sum()) for m in
                                          ("LightGBM (Tweedie)", "Endêmico-epidêmico (BN)", "Extra Trees")}
resumo["anos_LGBM_melhor_que_EE"] = int((val["LightGBM (Tweedie)"] < val["Endêmico-epidêmico (BN)"]).sum())

# ---------------------------------------------------------------------------
# 5. Calibração e captura de ondas
# ---------------------------------------------------------------------------
cal = []
for k in ("LightGBM", "Endêmico-epidêmico"):
    mu = pv[MODELOS[k]].values
    bordas = np.quantile(mu, np.linspace(0, 1, 21))
    faixa = np.clip(np.searchsorted(bordas, mu, side="right") - 1, 0, 19)
    g = pd.DataFrame({"faixa": faixa, "mu": mu, "y": yv}).groupby("faixa").agg(
        previsto=("mu", "mean"), observado=("y", "mean"), n=("y", "size"))
    g["modelo"] = k
    cal.append(g.reset_index())
cal = pd.concat(cal)
cal.to_csv(SAIDA / "a1b_calibracao.csv", index=False, sep=";", encoding="utf-8-sig")
resumo["calibracao_total"] = {k: float(pv[c].sum() / yv.sum()) for k, c in MODELOS.items()}

pv["onda"] = ((yv >= LIM_ONDA_MIN) & (yv >= LIM_ONDA_RR * pv["e"].values)).astype(int)
lift = []
fracs = np.r_[np.linspace(0.001, 0.02, 20), np.linspace(0.025, 0.2, 36)]
for k in ("LightGBM", "Endêmico-epidêmico", "Só população"):
    pv["prob"] = prob_onda(pv, pv[MODELOS[k]].values, pv["size_ee"].values)
    for f in fracs:
        cap = []
        for T, d in pv.groupby("ano"):
            n = int(np.ceil(f * len(d)))
            cap.append(d.sort_values("prob", ascending=False)["onda"].head(n).sum())
        lift.append(dict(modelo=k, fracao_monitorada=f, ondas_capturadas_pct=sum(cap) / pv["onda"].sum() * 100))
lift = pd.DataFrame(lift)
lift.to_csv(SAIDA / "a1b_captura_ondas.csv", index=False, sep=";", encoding="utf-8-sig")
for f in (0.01, 0.05, 0.10):  # fração de municípios monitorados mais próxima de 1%, 5% e 10%
    sub = lift.loc[lift.assign(d=(lift["fracao_monitorada"] - f).abs()).groupby("modelo")["d"].idxmin()]
    resumo[f"captura_{int(round(f * 100))}pct"] = dict(zip(sub["modelo"], sub["ondas_capturadas_pct"].round(1)))
resumo["total_ondas_2016_2022"] = int(pv["onda"].sum())
print({k: v for k, v in resumo.items() if k.startswith("captura") or k.startswith("calib")})

# ---------------------------------------------------------------------------
# 6. Sensibilidade aos hiperparâmetros (um de cada vez em torno da configuração usada)
# ---------------------------------------------------------------------------
VARIANTES = {
    "configuração usada": {},
    "folhas = 7": {"num_leaves": 7}, "folhas = 31": {"num_leaves": 31},
    "taxa de aprendizado = 0,1": {"learning_rate": 0.1},
    "mínimo por folha = 20": {"min_data_in_leaf": 20}, "mínimo por folha = 150": {"min_data_in_leaf": 150},
    "potência Tweedie = 1,2": {"tweedie_variance_power": 1.2},
    "potência Tweedie = 1,5": {"tweedie_variance_power": 1.5},
    "todas as variáveis e linhas": {"feature_fraction": 1.0, "bagging_fraction": 1.0, "bagging_freq": 0},
}
sens = []
for nome, mud in VARIANTES.items():
    ganhos = []
    for T in ANOS_TESTE:
        tr, te = dividir(T)
        mg, _ = treinar_lgbm(tr, ARVORE, {**PARAMS_LGBM, **mud})
        y = te["registros_lgbt"].values
        ganhos.append((1 - desvio_poisson(y, prever(mg, te, ARVORE)) / desvio_poisson(y, te["e"].values)) * 100)
    sens.append(dict(variante=nome, ganho_medio_pct=np.mean(ganhos), ganho_min_pct=np.min(ganhos),
                     ganho_max_pct=np.max(ganhos)))
    print(f"sensibilidade {nome}: {np.mean(ganhos):.1f}%")
sens = pd.DataFrame(sens)
sens.to_csv(SAIDA / "a1b_sensibilidade.csv", index=False, sep=";", encoding="utf-8-sig")
resumo["sensibilidade"] = sens.round(2).to_dict("records")

# ---------------------------------------------------------------------------
# 7. SHAP do modelo final (todos os anos), amostra estratificada para os gráficos
# ---------------------------------------------------------------------------
it = pd.read_csv(SAIDA / "a1_log_treinamento.csv", sep=";")["lgbm_arvores_escolhidas"]
final = lgb.train(PARAMS_LGBM, ds(p, ARVORE), int(np.median(it)))
amostra = pd.concat([p[p["registros_lgbt"] > 0].sample(2000, random_state=42),
                     p[p["registros_lgbt"] == 0].sample(2000, random_state=42)])
sv = shap.TreeExplainer(final).shap_values(amostra[ARVORE])
pd.DataFrame(sv, columns=ARVORE).to_csv(SAIDA / "a1b_shap_valores.csv", index=False, sep=";", encoding="utf-8-sig")
amostra[ARVORE + ["rr_hist", "pct_casais_mesmo_sexo_2022", "densidade", "y_lag3", "populacao"]].to_csv(
    SAIDA / "a1b_shap_amostra.csv", index=False, sep=";", encoding="utf-8-sig")
resumo["tempo_total_s"] = round(time.perf_counter() - t_inicio, 1)
json.dump(resumo, open(SAIDA / "a1b_resumo.json", "w"), ensure_ascii=False, indent=1, default=float)
print("tempo total (s):", resumo["tempo_total_s"])
