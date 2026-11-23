"""
Figuras do relatório (ACH0042 - Grupo 11), no padrão "figuras-relatorio-academico":
desenhadas no tamanho final (SBC, A4: texto de 15 cm), fonte serifada, paleta Okabe-Ito,
números em pt-BR, PDF vetorial com fontes TrueType embutidas.

Uso:  python figuras.py      (lê resultados/ e dados/; grava resultados/figuras/*.pdf)

Inclusão no LaTeX (a fração tem de ser a mesma usada em tamanho()):
  fig1_serie_mensal.pdf  -> width=\textwidth
  fig2_validacao.pdf     -> width=0.85\textwidth
  fig3_mapa_risco.pdf    -> width=0.62\textwidth
  fig4_perfil_trans.pdf  -> width=0.85\textwidth
  fig5_curvas_aprendizado.pdf, fig8_calibracao_captura.pdf, fig10_shap_dependencia.pdf -> width=\textwidth
  fig6_tamanho_treino.pdf, fig7_ablacao.pdf, fig9_shap_enxame.pdf                      -> width=0.85\textwidth
"""
import glob
import os
import urllib.request

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import font_manager
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

from comum import DADOS, SAIDA

# ---------------------------------------------------------------------------
# Padrão de figuras (skill figuras-relatorio-academico)
# ---------------------------------------------------------------------------
CM = 1 / 2.54
LARGURA_TEXTO_CM = 15.0  # \textwidth do SBC em A4

CORES = {
    "lgbt": "#0072B2", "trans": "#D55E00", "destaque": "#009E73",
    "secundaria": "#E69F00", "clara": "#56B4E9", "extra": "#CC79A7",
    "referencia": "#7F7F7F", "neutro": "#BDBDBD", "texto": "#222222",
}
CICLO = [CORES[k] for k in ("lgbt", "trans", "destaque", "secundaria", "clara", "extra")]
MESES = ["jan", "fev", "mar", "abr", "mai", "jun", "jul", "ago", "set", "out", "nov", "dez"]

# fontes TrueType locais opcionais (ex.: TeX Gyre Termes convertida), na pasta ./fontes
for _f in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fontes", "*.ttf")):
    font_manager.fontManager.addfont(_f)


def _fonte_serifada():
    disp = {f.name for f in font_manager.fontManager.ttflist}
    for nome in ("Termes TT", "TeX Gyre Termes", "Times New Roman", "Nimbus Roman", "STIXGeneral", "DejaVu Serif"):
        if nome in disp:
            return nome
    return "serif"


def aplicar_estilo():
    mpl.rcParams.update({
        "font.family": "serif", "font.serif": [_fonte_serifada()], "mathtext.fontset": "stix",
        "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.linewidth": 0.6, "axes.edgecolor": CORES["texto"], "axes.labelcolor": CORES["texto"],
        "xtick.color": CORES["texto"], "ytick.color": CORES["texto"],
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": "#E5E5E5", "grid.linewidth": 0.4,
        "axes.axisbelow": True, "axes.prop_cycle": mpl.cycler(color=CICLO),
        "lines.linewidth": 1.3, "lines.markersize": 4,
        "legend.frameon": False, "legend.handlelength": 1.8,
        "figure.dpi": 150, "savefig.dpi": 300, "figure.constrained_layout.use": True,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def tamanho(fracao=1.0, proporcao=0.62, altura_cm=None):
    """figsize em polegadas para ocupar `fracao` da largura do texto."""
    largura = LARGURA_TEXTO_CM * fracao
    altura = altura_cm if altura_cm else largura * proporcao
    return (largura * CM, altura * CM)


def num_br(casas=0, pct=False, sinal=False):
    """Formatador de eixo: 1.250 / 0,85 / 17,8% (sinal=True -> +16,7%)."""
    def f(x, _pos=None):
        s = f"{x:{'+' if sinal else ''},.{casas}f}".replace(",", "X").replace(".", ",").replace("X", ".")
        s = s.replace("-", "\u2212")  # sinal de menos tipográfico
        return s + ("%" if pct else "")
    return FuncFormatter(f)


def marcar_evento(ax, x, rotulo, y=1.0, ha="left"):
    """Linha vertical de evento com rótulo curto no topo (y em fração do eixo)."""
    ax.axvline(x, color=CORES["referencia"], ls=":", lw=0.8, zorder=1)
    dx = 2 if ha == "left" else -2
    ax.annotate(rotulo, xy=(x, y), xycoords=("data", "axes fraction"), xytext=(dx, -2),
                textcoords="offset points", ha=ha, va="top", fontsize=7, color=CORES["referencia"])


PASTA_FIG = SAIDA / "figuras"


def salvar(fig, nome, fracao, proporcional=False):
    """Salva o PDF garantindo que, após o corte de margens (bbox tight), a largura final seja
    exatamente `fracao` x 15 cm, sem ampliação no LaTeX (regra de ouro da skill)."""
    os.makedirs(PASTA_FIG, exist_ok=True)
    caminho = PASTA_FIG / (nome if nome.endswith(".pdf") else nome + ".pdf")
    alvo = LARGURA_TEXTO_CM * fracao * CM
    for _ in range(6):
        fig.canvas.draw()
        bb = fig.get_tightbbox(fig.canvas.get_renderer())
        largura = bb.width + 2 * mpl.rcParams["savefig.pad_inches"]
        if abs(largura - alvo) < 0.005:
            break
        w, h = fig.get_size_inches()
        if proporcional:  # mapas (aspecto fixo): escala largura e altura juntas
            fig.set_size_inches(w * alvo / largura, h * alvo / largura, forward=True)
        else:
            fig.set_size_inches(w + (alvo - largura), h, forward=True)
    fig.savefig(caminho)
    plt.close(fig)
    return caminho


aplicar_estilo()

# ---------------------------------------------------------------------------
# Figura 1 - série mensal (largura 1,0; proporção 0,42)
# ---------------------------------------------------------------------------
def fig1_serie_mensal():
    s = pd.read_csv(DADOS / "serie_mensal.csv", sep=";")
    s["data"] = pd.to_datetime(s["mes"])
    fig, ax = plt.subplots(figsize=tamanho(1.0, 0.42))
    ax.axvspan(pd.Timestamp("2020-03-01"), pd.Timestamp("2021-12-31"), color=CORES["neutro"], alpha=0.25, lw=0,
               zorder=0)
    ax.plot(s["data"], s["registros_lgbt"], color=CORES["lgbt"], label="Total LGBT")
    ax.plot(s["data"], s["registros_trans"], color=CORES["trans"], ls="--", label="Vítimas trans")
    ax.set_ylim(0, 760)
    marcar_evento(ax, pd.Timestamp("2019-06-13"), "Decisão do STF", y=1.0, ha="right")
    marcar_evento(ax, pd.Timestamp("2020-07-01"), "Novo sistema", y=0.88, ha="left")
    marcar_evento(ax, pd.Timestamp("2022-07-01"), "Nova ficha", y=1.0, ha="right")
    ax.annotate("Pandemia", xy=(pd.Timestamp("2021-02-01"), 0.70), xycoords=("data", "axes fraction"),
                ha="center", fontsize=7, color=CORES["referencia"])
    ax.annotate("sem registro trans", xy=(pd.Timestamp("2021-07-01"), 40), ha="center", va="center",
                fontsize=7, color=CORES["trans"])
    ax.set_xlim(pd.Timestamp("2010-12-01"), pd.Timestamp("2023-01-31"))
    ax.xaxis.set_major_locator(mpl.dates.YearLocator())
    ax.xaxis.set_major_formatter(mpl.dates.DateFormatter("%Y"))
    ax.yaxis.set_major_formatter(num_br())
    ax.set_ylabel("Denúncias por mês")
    ax.legend(loc="upper left", ncol=2)
    return salvar(fig, "fig1_serie_mensal", 1.0)


# ---------------------------------------------------------------------------
# Figura 2 - comparação de modelos (largura 0,85; proporção 0,45)
# ---------------------------------------------------------------------------
def fig2_validacao():
    v = pd.read_csv(SAIDA / "a1_validacao_resumo.csv", sep=";")
    nomes = {"LightGBM (Tweedie)": "LightGBM", "Endêmico-epidêmico (BN)": "Endêmico-epidêmico",
             "Extra Trees": "Extra Trees", "Só população": "Só população (referência)",
             "Participação histórica": "Fatia histórica da cidade", "Ano anterior": "Repetir o ano anterior"}
    modelos = {"LightGBM (Tweedie)", "Endêmico-epidêmico (BN)", "Extra Trees"}
    v["rotulo"] = v["modelo"].map(nomes)
    v = v.sort_values("ganho_desvio_vs_pop_%")
    fig, ax = plt.subplots(figsize=tamanho(0.85, 0.45))
    y = np.arange(len(v))
    cores = [CORES["lgbt"] if m in modelos else CORES["neutro"] for m in v["modelo"]]
    ax.barh(y, v["ganho_desvio_vs_pop_%"], color=cores, height=0.6, zorder=2)
    ax.axvline(0, color=CORES["referencia"], lw=0.8, zorder=3)
    fmt = num_br(1, pct=True, sinal=True)
    for yi, val, m in zip(y, v["ganho_desvio_vs_pop_%"], v["modelo"]):
        txt = "referência" if m == "Só população" else fmt(val)
        ax.text(val + (0.8 if val >= 0 else -0.8), yi, txt, va="center", ha="left" if val >= 0 else "right",
                fontsize=7.5, color=CORES["texto"])
    ax.set_yticks(y, v["rotulo"])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(num_br(0, pct=True))
    ax.set_xlim(-45, 25)
    ax.set_xlabel("Redução do erro de previsão em relação a “só população”")
    return salvar(fig, "fig2_validacao", 0.85)


# ---------------------------------------------------------------------------
# Figura 3 - mapa de risco relativo (largura 0,62; proporção ~1,1)
# ---------------------------------------------------------------------------
GEO = DADOS / "geo"
URL_MUN = "https://raw.githubusercontent.com/tbrugz/geodata-br/master/geojson/geojs-100-mun.json"
URL_UF = "https://raw.githubusercontent.com/codeforgermany/click_that_hood/main/public/data/brazil-states.geojson"


def _baixar(url, destino):
    if not destino.exists():
        os.makedirs(destino.parent, exist_ok=True)
        urllib.request.urlretrieve(url, destino)
    return destino


def fig3_mapa_risco():
    import geopandas as gpd

    mun = gpd.read_file(_baixar(URL_MUN, GEO / "municipios.geojson"))
    ufs = gpd.read_file(_baixar(URL_UF, GEO / "ufs.geojson"))
    r = pd.read_csv(SAIDA / "a1_risco_municipal.csv", sep=";", dtype={"cod_mun_ibge7": str})
    mun = mun.merge(r[["cod_mun_ibge7", "rr_eb_2020_2022", "hotspot_2020_2022"]], left_on="id",
                    right_on="cod_mun_ibge7", how="left")
    mun["log_rr"] = np.log2(mun["rr_eb_2020_2022"].clip(1 / 16, 16))

    fig, ax = plt.subplots(figsize=tamanho(0.62, 1.1))
    norm = TwoSlopeNorm(vmin=-4, vcenter=0, vmax=4)  # log2(RR): 1/16 ... 1 ... 16
    sem = mun[mun["log_rr"].isna()]
    com = mun[mun["log_rr"].notna()]
    sem.plot(ax=ax, color="#F0F0F0", linewidth=0, rasterized=True)
    com.plot(ax=ax, column="log_rr", cmap="RdBu_r", norm=norm, linewidth=0, rasterized=True)
    ufs.boundary.plot(ax=ax, color=CORES["referencia"], linewidth=0.3)
    hs = mun[mun["hotspot_2020_2022"] == 1]
    hs.boundary.plot(ax=ax, color="black", linewidth=0.6)
    ax.set_axis_off()

    sm = mpl.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cb = fig.colorbar(sm, ax=ax, orientation="horizontal", shrink=0.6, aspect=30, pad=0.02)
    ticks = [-4, -2, 0, 2, 4]
    cb.set_ticks(ticks)
    cb.set_ticklabels(["1/16", "1/4", "1", "4", "16"])
    cb.ax.tick_params(labelsize=8, width=0.6, length=2)
    cb.outline.set_linewidth(0.4)
    cb.set_label("Risco relativo suavizado (1 = esperado para o porte)", fontsize=8)
    ax.legend(handles=[Patch(facecolor="none", edgecolor="black", linewidth=0.6,
                             label=f"Aglomerado de alto risco (n = {len(hs)})"),
                       Patch(facecolor="#F0F0F0", edgecolor="none", label="Sem dados")],
              loc="lower left", fontsize=7, handlelength=1.2)
    return salvar(fig, "fig3_mapa_risco", 0.62, proporcional=True)


# ---------------------------------------------------------------------------
# Figura 4 - forest plot do perfil das vítimas trans (largura 0,85)
# ---------------------------------------------------------------------------
def fig4_perfil_trans():
    t = pd.read_csv(SAIDA / "a3_logistica_razoes_chance.csv", sep=";", index_col=0)
    blocos = [
        ("Tipo de violação", {"viol_vida": "Contra a vida", "viol_sexual": "Sexual", "viol_fisica": "Física",
                              "viol_institucional": "Institucional", "viol_psicologica": "Psicológica",
                              "viol_discriminacao": "Discriminação"}),
        ("Local (ref.: residência)", {"cenario_via_publica_transporte": "Via pública ou transporte",
                                      "cenario_instituicao_publica_ou_religiosa": "Instituição pública ou religiosa",
                                      "cenario_trabalho": "Trabalho"}),
        ("Agressor (ref.: familiar)", {"relacao_suspeito_institucional_profissional": "Profissional ou institucional",
                                       "relacao_suspeito_trabalho": "Relação de trabalho",
                                       "relacao_suspeito_desconhecido_da_vitima": "Desconhecido da vítima"}),
        ("Idade (ref.: 18 a 24 anos)", {"faixa_etaria_25-29": "25 a 29 anos", "faixa_etaria_0-17": "0 a 17 anos",
                                        "faixa_etaria_60+": "60 anos ou mais"}),
        ("Registro", {"raca_cor_desconhecido": "Raça/cor não informada"}),
    ]
    linhas, rotulos, cabecalhos = [], [], []
    pos = 0
    for titulo, itens in blocos:
        cabecalhos.append((pos, titulo))
        pos += 1
        for var, rot in itens.items():
            linhas.append((pos, t.loc[var]))
            rotulos.append((pos, rot))
            pos += 1
    n = pos
    fig, ax = plt.subplots(figsize=tamanho(0.85, altura_cm=0.42 * n + 1.5))
    for p, row in linhas:
        sig = row["p"] < 0.05
        cor = CORES["trans"] if sig else CORES["referencia"]
        ax.plot([row["ic95_inf"], row["ic95_sup"]], [p, p], color=cor, lw=1.0, solid_capstyle="butt")
        ax.plot(row["razao_chances"], p, marker="s", ms=4.5, ls="none", zorder=3, color=cor,
                mfc=cor if sig else "white", mec=cor, mew=0.9)  # vazado = não significativo (legível em P&B)
    ax.axvline(1, color=CORES["referencia"], ls="--", lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([0.25, 0.5, 1, 2, 4]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: {0.25: "0,25", 0.5: "0,5", 1: "1", 2: "2", 4: "4"}
                                               .get(round(x, 2), "")))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(0.12, 5)
    ax.set_ylim(n - 0.5, -0.7)
    ax.set_yticks([p for p, _ in rotulos], [r for _, r in rotulos])
    ax.tick_params(axis="y", length=0)
    # títulos dos blocos alinhados à esquerda da coluna de rótulos (itens ficam recuados)
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    larg_pt = max(lb.get_window_extent(rend).width for lb in ax.get_yticklabels()) * 72 / fig.dpi
    for p, titulo in cabecalhos:
        ax.annotate(titulo, xy=(0, p), xycoords=("axes fraction", "data"), xytext=(-(larg_pt + 10), 0),
                    textcoords="offset points", ha="left", va="center", fontsize=8, fontweight="bold",
                    color=CORES["texto"], annotation_clip=False)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True, which="major")
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Razão de chances de a vítima ser trans (IC 95%, escala log)")
    ax.legend(handles=[Line2D([], [], color=CORES["trans"], marker="s", ms=4.5, lw=1.0, label="p < 0,05"),
                       Line2D([], [], color=CORES["referencia"], marker="s", ms=4.5, lw=1.0, mfc="white",
                              mew=0.9, label="p ≥ 0,05")],
              loc="lower right", fontsize=7)
    return salvar(fig, "fig4_perfil_trans", 0.85)


# ---------------------------------------------------------------------------
# Aprofundamento da Análise 1 (a1b_aprendizado.py): figuras 5 a 10
# ---------------------------------------------------------------------------
COR_MODELO = {"LightGBM": CORES["lgbt"], "Endêmico-epidêmico": CORES["destaque"], "Só população": CORES["referencia"]}
ROTULO_VAR = {
    "l_rr_hist": "Histórico suavizado da cidade", "pct_casais_mesmo_sexo_2022": "Casais do mesmo sexo (%)",
    "log_dens": "Densidade (log)", "l_lag3": "Denúncias nos 3 anos anteriores", "pct_pretos_pardos": "Pretos e pardos (%)",
    "log_pop": "População (log)", "l_lag_viz": "Denúncias dos vizinhos (ano anterior)", "reg_Sul": "Região Sul",
    "l_lag_esc": "Denúncias no ano anterior (ajustadas)", "pct_15_29": "População de 15 a 29 anos (%)",
    "log_hom_lag": "Homicídios (log, ano anterior)", "gini_2010": "Índice de Gini", "log_renda": "Renda (log)",
    "log_pib_lag": "PIB per capita (log, ano anterior)", "l_lag": "Denúncias no ano anterior",
    "munic_indice_lgbt": "Índice LGBT (MUNIC)", "munic_ouvidoria_dh": "Ouvidoria de direitos humanos",
    "munic_estrutura_dh": "Estrutura de direitos humanos", "processo_transexualizador": "Processo Transexualizador",
    "log_proc_pt": "Procedimentos do Proc. Transexualizador", "capital": "Capital", "reg_Nordeste": "Região Nordeste",
    "reg_Norte": "Região Norte", "reg_Centro-Oeste": "Região Centro-Oeste",
}


def letra(ax, txt):
    ax.text(-0.02, 1.04, txt, transform=ax.transAxes, fontsize=10, fontweight="bold", ha="right", va="bottom")


def fig5_curvas_aprendizado():
    c = pd.read_csv(SAIDA / "a1b_curvas_aprendizado.csv", sep=";")
    fig, (a, b) = plt.subplots(1, 2, figsize=tamanho(1.0, 0.42))
    d = c[c["ano_previsto"] == 2022]
    parada = d["rodada_escolhida"].iat[0]
    a.plot(d["rodada"], d["perda_treino"], color=CORES["referencia"], ls="--", label="Treino (2012–2020)")
    a.plot(d["rodada"], d["perda_validacao"], color=CORES["lgbt"], label="Validação (2021)")
    a.axvline(parada, color=CORES["texto"], ls=":", lw=0.8)
    a.annotate(f"parada: {parada} árvores", xy=(parada, 0.97), xycoords=("data", "axes fraction"), xytext=(3, 0),
               textcoords="offset points", fontsize=7, va="top", color=CORES["texto"])
    a.set_xlabel("Árvores adicionadas (rodadas)")
    a.set_ylabel("Perda Tweedie (menor = melhor)")
    a.yaxis.set_major_formatter(num_br(2))
    a.xaxis.set_major_formatter(num_br())
    a.legend(loc="center right")
    letra(a, "(a)")
    anos = sorted(c["ano_previsto"].unique())
    cmap = mpl.colormaps["cividis"]
    for i, ano in enumerate(anos):
        d = c[c["ano_previsto"] == ano]
        rel = (d["perda_validacao"] / d["perda_validacao"].min() - 1) * 100
        cor = cmap(i / (len(anos) - 1))
        b.plot(d["rodada"], rel, color=cor, lw=1.0, label=str(ano))
        pr = d["rodada_escolhida"].iat[0]
        b.plot(pr, rel.iat[pr - 1], marker="o", ms=3.5, color=cor, mec="white", mew=0.5, zorder=3)
    b.set_ylim(-0.5, 14)
    b.set_xlabel("Árvores adicionadas (rodadas)")
    b.set_ylabel("Perda de validação acima do mínimo")
    b.yaxis.set_major_formatter(num_br(0, pct=True))
    b.xaxis.set_major_formatter(num_br())
    b.legend(title="Ano previsto", ncol=2, loc="upper right", fontsize=7, title_fontsize=7, handlelength=1.2)
    letra(b, "(b)")
    return salvar(fig, "fig5_curvas_aprendizado", 1.0)


def fig6_tamanho_treino():
    t = pd.read_csv(SAIDA / "a1b_tamanho_treino.csv", sep=";")
    t = t[t["anos_treino"] >= 2]  # com 1 ano não há ano reservado para a parada antecipada do LightGBM
    fig, eixos = plt.subplots(1, 2, figsize=tamanho(0.85, 0.48), sharey=False)
    for ax, ano, lt in zip(eixos, (2019, 2022), ("(a)", "(b)")):
        d = t[t["ano_previsto"] == ano]
        for mod, ls, mk in (("LightGBM", "-", "o"), ("Endêmico-epidêmico", "--", "s")):
            e = d[d["modelo"] == mod]
            ax.plot(e["anos_treino"], e["ganho_vs_pop_pct"], color=COR_MODELO[mod], ls=ls, marker=mk, ms=3.5,
                    label=mod)
        ax.axhline(0, color=CORES["referencia"], lw=0.8)
        ax.set_xlabel("Anos usados no treino")
        ax.set_xticks(range(2, d["anos_treino"].max() + 1))
        ax.yaxis.set_major_formatter(num_br(0, pct=True))
        ax.set_title(f"Previsão de {ano}", fontsize=9)
        letra(ax, lt)
    eixos[0].set_ylabel("Redução do erro vs. só população")
    eixos[1].legend(loc="lower right")
    return salvar(fig, "fig6_tamanho_treino", 0.85)


def fig7_ablacao():
    r = pd.read_csv(SAIDA / "a1b_ablacao_resumo.csv", sep=";")
    ordem = ["Só estrutura", "Só histórico da cidade", "Estrutura + histórico", "Completo (+ vizinhos)"]
    r = r.set_index("conjunto").loc[ordem[::-1]].reset_index()
    pop = pd.read_csv(SAIDA / "a1_validacao_resumo.csv", sep=";").set_index("modelo").loc["Só população", "prec20_onda"]
    fig, (a, b) = plt.subplots(1, 2, figsize=tamanho(0.85, 0.42), sharey=True)
    y = np.arange(len(r))
    cores = [CORES["lgbt"] if c == "Completo (+ vizinhos)" else CORES["clara"] for c in r["conjunto"]]
    a.barh(y, r["ganho_vs_pop_pct"], color=cores, height=0.6)
    for yi, v in zip(y, r["ganho_vs_pop_pct"]):
        a.text(v + 0.3, yi, num_br(1, pct=True)(v), va="center", fontsize=7.5)
    a.set_xlim(0, 21)
    a.set_xlabel("Redução do erro vs. só população")
    a.xaxis.set_major_formatter(num_br(0, pct=True))
    b.barh(y, r["prec20_onda"] * 100, color=cores, height=0.6)
    b.axvline(pop * 100, color=CORES["referencia"], ls="--", lw=0.8)
    b.annotate("só população", xy=(pop * 100, 1.0), xycoords=("data", "axes fraction"), xytext=(0, 2),
               textcoords="offset points", fontsize=7, va="bottom", ha="center", color=CORES["referencia"],
               annotation_clip=False)
    for yi, v in zip(y, r["prec20_onda"] * 100):
        b.text(v + 0.6, yi, num_br(0, pct=True)(v), va="center", fontsize=7.5)
    b.set_xlim(0, 48)
    b.set_xlabel("Acerto entre as 20 cidades apontadas")
    b.xaxis.set_major_formatter(num_br(0, pct=True))
    a.set_yticks(y, r["conjunto"])
    for ax, lt in ((a, "(a)"), (b, "(b)")):
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.tick_params(axis="y", length=0)
        letra(ax, lt)
    return salvar(fig, "fig7_ablacao", 0.85)


def fig8_calibracao_captura():
    c = pd.read_csv(SAIDA / "a1b_calibracao.csv", sep=";")
    l = pd.read_csv(SAIDA / "a1b_captura_ondas.csv", sep=";")
    fig, (a, b) = plt.subplots(1, 2, figsize=tamanho(1.0, 0.42))
    lim = (0.003, 120)
    a.plot(lim, lim, color=CORES["referencia"], ls="--", lw=0.8)
    # a diagonal tracejada (previsto = observado) é explicada na legenda da figura no LaTeX
    for mod, mk in (("LightGBM", "o"), ("Endêmico-epidêmico", "s")):
        d = c[c["modelo"] == mod]
        a.plot(d["previsto"], d["observado"].clip(lower=lim[0]), marker=mk, ms=3.5, lw=1.0,
               ls="-" if mod == "LightGBM" else "--", color=COR_MODELO[mod], label=mod)
    a.set_xscale("log"), a.set_yscale("log")
    a.set_xlim(*lim), a.set_ylim(*lim)
    fmt = FuncFormatter(lambda x, _: num_br(3 if x < 0.01 else 2 if x < 0.1 else 1 if x < 1 else 0)(x))
    a.xaxis.set_major_formatter(fmt), a.yaxis.set_major_formatter(fmt)
    a.set_xlabel("Denúncias previstas (média da faixa)")
    a.set_ylabel("Denúncias observadas (média da faixa)")
    a.legend(loc="upper left")
    letra(a, "(a)")
    for mod, ls in (("LightGBM", "-"), ("Endêmico-epidêmico", "--"), ("Só população", ":")):
        d = l[l["modelo"] == mod]
        b.plot(d["fracao_monitorada"] * 100, d["ondas_capturadas_pct"], color=COR_MODELO[mod], ls=ls, label=mod)
    b.plot([0, 20], [0, 20], color=CORES["neutro"], lw=0.8, label="Ao acaso")
    b.set_xlim(0, 20), b.set_ylim(0, 100)
    b.xaxis.set_major_formatter(num_br(0, pct=True)), b.yaxis.set_major_formatter(num_br(0, pct=True))
    b.set_xlabel("Municípios monitorados (os de maior risco previsto)")
    b.set_ylabel("Ondas capturadas")
    b.legend(loc="lower right")
    letra(b, "(b)")
    return salvar(fig, "fig8_calibracao_captura", 1.0)


def _ordem_shap(k=10):
    sv = pd.read_csv(SAIDA / "a1b_shap_valores.csv", sep=";")
    imp = sv.abs().mean().sort_values(ascending=False)
    return sv, imp.index[:k].tolist()


def fig9_shap_enxame():
    sv, top = _ordem_shap(10)
    am = pd.read_csv(SAIDA / "a1b_shap_amostra.csv", sep=";")
    fig, ax = plt.subplots(figsize=tamanho(0.85, altura_cm=0.62 * len(top) + 2.2))
    rng_ = np.random.default_rng(0)
    for i, var in enumerate(top[::-1]):
        x = sv[var].values
        v = am[var].rank(pct=True).values
        dens = np.histogram(x, bins=60)[0]
        idx = np.clip(np.digitize(x, np.histogram_bin_edges(x, 60)) - 1, 0, 59)
        jit = (rng_.random(len(x)) - 0.5) * 0.75 * (dens[idx] / dens.max()) ** 0.5
        sc = ax.scatter(x, i + jit, c=v, cmap="cividis", s=2.2, lw=0, rasterized=True, vmin=0, vmax=1)
    ax.axvline(0, color=CORES["referencia"], lw=0.8)
    ax.set_yticks(range(len(top)), [ROTULO_VAR.get(v, v) for v in top[::-1]])
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False), ax.grid(axis="x", visible=True)
    ax.xaxis.set_major_formatter(num_br(1))
    ax.set_xlabel("Contribuição para a previsão (SHAP; +0,7 ≈ dobra o esperado)")
    cb = fig.colorbar(sc, ax=ax, orientation="vertical", shrink=0.7, aspect=30, pad=0.02)
    cb.set_ticks([0, 1]), cb.set_ticklabels(["baixo", "alto"])
    cb.set_label("Valor da variável", fontsize=8)
    cb.ax.tick_params(labelsize=7, length=0)
    cb.outline.set_linewidth(0.4)
    return salvar(fig, "fig9_shap_enxame", 0.85)


def fig10_shap_dependencia():
    sv, _ = _ordem_shap()
    am = pd.read_csv(SAIDA / "a1b_shap_amostra.csv", sep=";")
    paineis = [("l_rr_hist", "rr_hist", "Risco histórico suavizado da cidade", True),
               ("pct_casais_mesmo_sexo_2022", "pct_casais_mesmo_sexo_2022", "Casais do mesmo sexo (%)", False),
               ("log_dens", "densidade", "Densidade (hab./km²)", True)]
    fig, eixos = plt.subplots(1, 3, figsize=tamanho(1.0, 0.36), sharey=True)
    for ax, (var, orig, rot, log), lt in zip(eixos, paineis, ("(a)", "(b)", "(c)")):
        x, y = am[orig].values, sv[var].values
        ax.scatter(x, y, s=2, color=CORES["lgbt"], alpha=0.25, lw=0, rasterized=True)
        q = pd.qcut(np.log(x) if log else x, 20, duplicates="drop")
        m = pd.DataFrame({"x": x, "y": y, "q": q}).groupby("q", observed=True).median()
        ax.plot(m["x"], m["y"], color=CORES["trans"], lw=1.3)
        ax.axhline(0, color=CORES["referencia"], lw=0.8)
        if log:
            ax.set_xscale("log")
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: num_br(1 if v < 1 else 0)(v)))
        ax.set_xlabel(rot)
        letra(ax, lt)
    eixos[0].set_ylabel("Contribuição (SHAP)")
    eixos[0].yaxis.set_major_formatter(num_br(1))
    return salvar(fig, "fig10_shap_dependencia", 1.0)


if __name__ == "__main__":
    for f in (fig1_serie_mensal, fig2_validacao, fig3_mapa_risco, fig4_perfil_trans, fig5_curvas_aprendizado,
              fig6_tamanho_treino, fig7_ablacao, fig8_calibracao_captura, fig9_shap_enxame, fig10_shap_dependencia):
        print(f())
