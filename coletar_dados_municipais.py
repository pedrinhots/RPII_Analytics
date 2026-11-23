"""
Coleta dos indicadores municipais (IBGE/SIDRA, Ipea, MUNIC, Atlas Brasil)
ACH0042 - Grupo 11 - EACH-USP

Uso (Anaconda Prompt ou terminal):
    python coletar_dados_municipais.py

Cria a pasta ./dados_municipais/ com:
    painel_anual.csv        município x ano (2011-2022): população, PIB, homicídios
    censo_2010_2022.csv     1 linha por município: indicadores dos Censos 2010 e 2022
    municipios.csv          lista dos 5.570 municípios (código, nome, UF, região, capital)
    brutos/                 arquivos originais da MUNIC (2014, 2019, 2023) e do Atlas Brasil
    log_coleta.txt          o que deu certo e o que falhou

Leva de 5 a 15 minutos (a API do SIDRA é lenta). Pode rodar de novo: o que já foi
baixado é reaproveitado.
Dependências: pandas, requests (já vêm no Anaconda).
"""
import json
import time
from pathlib import Path

import pandas as pd
import requests

SAIDA = Path("dados_municipais")
CACHE = SAIDA / "cache"
BRUTOS = SAIDA / "brutos"
ANOS = list(range(2011, 2023))
SIDRA = "https://apisidra.ibge.gov.br/values"
LOG = []

for p in (SAIDA, CACHE, BRUTOS):
    p.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(msg)
    LOG.append(msg)


def get_json(url, nome_cache, tentativas=4):
    """GET com cache em disco e novas tentativas (a API do SIDRA cai às vezes)."""
    arq = CACHE / f"{nome_cache}.json"
    if arq.exists():
        return json.loads(arq.read_text(encoding="utf-8"))
    for i in range(tentativas):
        try:
            r = requests.get(url, timeout=180)
            r.raise_for_status()
            dados = r.json()
            arq.write_text(json.dumps(dados), encoding="utf-8")
            return dados
        except Exception as e:  # noqa: BLE001
            espera = 10 * (i + 1)
            log(f"  falha em {nome_cache} ({e}); nova tentativa em {espera}s")
            time.sleep(espera)
    raise RuntimeError(f"não foi possível baixar {nome_cache}")


def sidra(caminho, nome):
    """Consulta a API do SIDRA e devolve DataFrame com colunas legíveis."""
    dados = get_json(SIDRA + caminho, nome)
    cab, linhas = dados[0], dados[1:]
    df = pd.DataFrame(linhas).rename(columns=cab)
    # convenção do SIDRA: "-" = zero absoluto; "..", "...", "X" = não aplicável / indisponível / sigilo
    df["Valor"] = pd.to_numeric(df["Valor"].replace("-", "0"), errors="coerce")
    return df


def baixar_arquivo(url, destino):
    destino = BRUTOS / destino
    if destino.exists() and destino.stat().st_size > 0:
        return
    for i in range(3):
        try:
            with requests.get(url, timeout=600, stream=True) as r:
                r.raise_for_status()
                with open(destino, "wb") as f:
                    for bloco in r.iter_content(1 << 20):
                        f.write(bloco)
            log(f"  ok: {destino.name} ({destino.stat().st_size / 1e6:.1f} MB)")
            return
        except Exception as e:  # noqa: BLE001
            log(f"  falha em {destino.name} ({e})")
            time.sleep(10)


# ---------------------------------------------------------------------------
def municipios():
    log("1/6 Lista de municípios (API de localidades do IBGE)")
    dados = get_json("https://servicodados.ibge.gov.br/api/v1/localidades/municipios", "localidades")
    capitais = {"1100205", "1302603", "1200401", "5002704", "1600303", "5300108", "1400100", "5103403",
                "1721000", "3550308", "2211001", "3304557", "1501402", "5208707", "2927408", "4205407",
                "2111300", "2704302", "4314902", "4106902", "3106200", "2304400", "2611606", "2507507",
                "2800308", "2408102", "3205309"}
    linhas = []
    for m in dados:
        uf = m["microrregiao"]["mesorregiao"]["UF"] if m.get("microrregiao") else m["regiao-imediata"]["regiao-intermediaria"]["UF"]
        linhas.append({"cod_mun_ibge7": str(m["id"]), "cod_mun_ibge6": str(m["id"])[:6], "municipio": m["nome"],
                       "uf": uf["sigla"], "regiao": uf["regiao"]["nome"], "capital": int(str(m["id"]) in capitais)})
    df = pd.DataFrame(linhas)
    df.to_csv(SAIDA / "municipios.csv", index=False, sep=";", encoding="utf-8-sig")
    log(f"  {len(df)} municípios")
    return df


def painel_anual(mun):
    log("2/6 Painel anual: população (SIDRA 6579 + Censo 2022 / 4714) e PIB (SIDRA 5938)")
    pop = []
    for ano in ANOS:
        if ano == 2022:
            df = sidra("/t/4714/n6/all/v/93/p/2022", "pop_2022_censo")
        else:
            df = sidra(f"/t/6579/n6/all/v/9324/p/{ano}", f"pop_{ano}")
        pop.append(df[["Município (Código)", "Valor"]].assign(ano=ano))
        print(f"   população {ano}: {len(df)}")
    pop = pd.concat(pop).rename(columns={"Município (Código)": "cod_mun_ibge7", "Valor": "populacao"})

    pib = []
    for ano in ANOS:
        try:
            df = sidra(f"/t/5938/n6/all/v/37/p/{ano}", f"pib_{ano}")
            pib.append(df[["Município (Código)", "Valor"]].assign(ano=ano))
            print(f"   PIB {ano}: {len(df)}")
        except RuntimeError as e:
            log(f"  PIB {ano} indisponível: {e}")
    pib = pd.concat(pib).rename(columns={"Município (Código)": "cod_mun_ibge7", "Valor": "pib_mil_reais"})

    log("3/6 Homicídios registrados por município (Ipea, Atlas da Violência, série 328)")
    hom = get_json("https://www.ipea.gov.br/dados-api/series-values/328/4", "ipea_homicidios_328")
    hom = pd.DataFrame(hom)
    hom = hom.assign(cod_mun_ibge6=hom["regiao_id"].astype(str).str[:6], ano=hom["periodo"].str[:4].astype(int),
                     homicidios=pd.to_numeric(hom["valor"], errors="coerce"))[["cod_mun_ibge6", "ano", "homicidios"]]

    base = mun[["cod_mun_ibge7", "cod_mun_ibge6"]].merge(pd.DataFrame({"ano": ANOS}), how="cross")
    painel = (base.merge(pop, on=["cod_mun_ibge7", "ano"], how="left")
                  .merge(pib, on=["cod_mun_ibge7", "ano"], how="left")
                  .merge(hom, on=["cod_mun_ibge6", "ano"], how="left"))
    # a série do Ipea só lista municípios com ao menos 1 homicídio no ano -> ausente = 0
    painel["homicidios"] = painel["homicidios"].fillna(0)
    painel["pib_per_capita"] = painel["pib_mil_reais"] * 1000 / painel["populacao"]
    painel["taxa_homicidios_100mil"] = painel["homicidios"] / painel["populacao"] * 1e5
    painel.to_csv(SAIDA / "painel_anual.csv", index=False, sep=";", encoding="utf-8-sig")
    log(f"  painel: {painel.shape}")


def censos(mun):
    log("4/6 Indicadores dos Censos 2010 e 2022 (SIDRA)")
    out = mun[["cod_mun_ibge7"]].copy()

    def pivot(df, col_cat, mapa, prefixo):
        df = df[df[col_cat].isin(mapa)]
        df = df.assign(chave=prefixo + df[col_cat].map(mapa) + "_" + df["Ano"].astype(str))
        return df.pivot_table(index="Município (Código)", columns="chave", values="Valor", aggfunc="first")

    # densidade, área (Censo 2022)
    d = sidra("/t/4714/n6/all/v/93,6318,614/p/2022", "censo2022_densidade")
    d = d.pivot_table(index="Município (Código)", columns="Variável", values="Valor", aggfunc="first")
    d.columns = ["area_km2_2022" if "Área" in c else "densidade_2022" if "Densidade" in c else "populacao_2022" for c in d.columns]
    out = out.merge(d, left_on="cod_mun_ibge7", right_index=True, how="left")

    # cor ou raça (2010 e 2022)
    c = sidra("/t/9605/n6/all/v/93/p/2010,2022/c86/95251,2777,2779", "censo_cor")
    c = pivot(c, "Cor ou raça", {"Total": "total", "Preta": "preta", "Parda": "parda"}, "pop_")
    out = out.merge(c, left_on="cod_mun_ibge7", right_index=True, how="left")
    for a in (2010, 2022):
        out[f"pct_pretos_pardos_{a}"] = (out[f"pop_preta_{a}"] + out[f"pop_parda_{a}"]) / out[f"pop_total_{a}"] * 100

    # jovens 15-29 (2010 e 2022)
    j = sidra("/t/9606/n6/all/v/93/p/2010,2022/c86/95251/c2/6794/c287/100362,93086,93087,93088", "censo_idade")
    j = pivot(j, "Idade", {"Total": "total", "15 a 19 anos": "15_19", "20 a 24 anos": "20_24", "25 a 29 anos": "25_29"}, "idade_")
    for a in (2010, 2022):
        j[f"pct_15_29_{a}"] = (j[f"idade_15_19_{a}"] + j[f"idade_20_24_{a}"] + j[f"idade_25_29_{a}"]) / j[f"idade_total_{a}"] * 100
    out = out.merge(j[[f"pct_15_29_{a}" for a in (2010, 2022)]], left_on="cod_mun_ibge7", right_index=True, how="left")

    # renda domiciliar per capita (2010: 3578; 2022: 10295)
    r10 = sidra("/t/3578/n6/all/v/2012/p/2010/c1/6795/c386/9680", "censo2010_renda")
    r22 = sidra("/t/10295/n6/all/v/13431/p/2022/c2/6794/c86/95251/c58/95253", "censo2022_renda")
    for df, nome in ((r10, "renda_pc_2010"), (r22, "renda_pc_2022")):
        out = out.merge(df.set_index("Município (Código)")["Valor"].rename(nome), left_on="cod_mun_ibge7",
                        right_index=True, how="left")

    # casais do mesmo sexo (2022)
    s = sidra("/t/9881/n6/all/v/800/p/2022/c1885/72599,72601", "censo2022_casais")
    s = s.pivot_table(index="Município (Código)", columns="Presença de cônjuge da pessoa responsável pelo domicílio",
                      values="Valor", aggfunc="first")
    s.columns = ["domicilios_mesmo_sexo_2022" if "mesmo sexo" in c else "domicilios_total_2022" for c in s.columns]
    s["pct_casais_mesmo_sexo_2022"] = s["domicilios_mesmo_sexo_2022"] / s["domicilios_total_2022"] * 100
    out = out.merge(s, left_on="cod_mun_ibge7", right_index=True, how="left")

    out = mun.merge(out, on="cod_mun_ibge7", how="left")
    out.to_csv(SAIDA / "censo_2010_2022.csv", index=False, sep=";", encoding="utf-8-sig")
    log(f"  censo: {out.shape}; ausentes por coluna (%):")
    log("  " + (out.isna().mean() * 100).round(1)[lambda s: s > 0].to_string().replace("\n", "\n  "))


def arquivos_brutos():
    log("5/6 MUNIC (Direitos Humanos) e Atlas Brasil (IDHM, Gini 2010)")
    baixar_arquivo("https://ftp.ibge.gov.br/Perfil_Municipios/2014/base_MUNIC_xls_2014.zip", "MUNIC_2014.zip")
    baixar_arquivo("https://ftp.ibge.gov.br/Perfil_Municipios/2019/Base_de_Dados/Base_MUNIC_2019_20210817.xlsx", "MUNIC_2019.xlsx")
    baixar_arquivo("https://ftp.ibge.gov.br/Perfil_Municipios/2023/Base_de_Dados/Base_MUNIC_2023.xlsx", "MUNIC_2023.xlsx")
    baixar_arquivo("https://www.atlasbrasil.org.br/cockpit/storage/uploads/dados/censo_total_1991_2010.xlsx", "atlas_censo_1991_2010.xlsx")
    baixar_arquivo("https://www.atlasbrasil.org.br/cockpit/storage/uploads/dados/censo_metadados_1991_2010.xlsx", "atlas_metadados.xlsx")


def main():
    t0 = time.time()
    etapas = [("municípios", None), ("painel", painel_anual), ("censos", censos), ("brutos", arquivos_brutos)]
    mun = None
    for nome, func in etapas:
        try:
            if nome == "municípios":
                mun = municipios()
            elif nome == "brutos":
                func()
            else:
                func(mun)
        except Exception as e:  # noqa: BLE001
            log(f"ERRO na etapa {nome}: {e!r}")
    log(f"6/6 Concluído em {(time.time() - t0) / 60:.1f} min. Arquivos em: {SAIDA.resolve()}")
    (SAIDA / "log_coleta.txt").write_text("\n".join(LOG), encoding="utf-8")


if __name__ == "__main__":
    main()