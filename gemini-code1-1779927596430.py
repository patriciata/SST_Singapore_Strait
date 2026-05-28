# Codigo 01 - ok ================================================
# Adaptado para rodar em repositório local/GitHub (sem Google Drive)
# Rodar 2x: para month=7 e para month=12
# ===================== CONFIG (apenas L3m) =====================

# 1. MUDANÇA: O caminho raiz agora é a pasta atual (onde o script está rodando)
# Garanta que a pasta "Dados/DAY/L3m" exista junto com o seu script no GitHub
ROOT = "./Dados"
L3M_DIR = f"{ROOT}/DAY/L3m"     # só L3m
ANOS = (2002, 2025)             # intervalo inclusivo
MONTH = 12                      # << mês desejado: 1..12 (1=jan, 7=jul, 12=dez)
COMPRESS_GZ = True              # salva como .csv.gz; se quiser .csv normal: False
VARS_TO_KEEP = None             # ex.: ["sst"] para exportar só TSM (opcional)

# Recorte geográfico do Estreito de Cingapura (lon_min, lon_max, lat_min, lat_max)
BBOX = (103.315900, 104.557400, 0.904561, 1.549800)
# ===============================================================

# 2. MUDANÇA: A montagem do Google Drive foi removida.
# O GitHub ou o PC local não precisam (nem conseguem) usar a biblioteca google.colab.
print("↪ Executando em ambiente local/GitHub (sem necessidade de Drive).")

import os, glob, re, traceback
import numpy as np
import pandas as pd
import xarray as xr
from datetime import datetime

print("\n▶ Passo 1/6: Preparando utilitários...")

def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path

def parse_date_from_name(fname: str):
    """Extrai AAAAMMDD do nome (formato Aqua-MODIS)."""
    m = re.search(r"\.(\d{8})\.", os.path.basename(fname))
    if not m: return None
    s = m.group(1)
    try:
        return datetime(int(s[:4]), int(s[4:6]), int(s[6:8]))
    except Exception:
        return None

def _parse_any_datetime(s: str):
    if not s: return None
    s = str(s).strip().replace("Z","")
    try:
        return pd.to_datetime(s, utc=False, errors="raise").to_pydatetime()
    except Exception:
        return None

def extract_datetime(ds: xr.Dataset, fname: str):
    """Prioriza attrs → variável time → nome do arquivo."""
    for k in ["time_coverage_start", "time_start", "start_time", "date_created"]:
        dt = _parse_any_datetime(ds.attrs.get(k))
        if dt: return dt
    if "time" in ds.variables:
        try:
            tvar = ds["time"]
            if tvar.ndim == 0:
                dt = pd.to_datetime(pd.Index([tvar.values]), errors="coerce")[0]
            else:
                dt = pd.to_datetime(pd.Index([tvar.values[0]]), errors="coerce")[0]
            if pd.notnull(dt): return pd.Timestamp(dt).to_pydatetime()
        except Exception:
            pass
    return parse_date_from_name(fname)

def open_ds_any(f):
    try:
        return xr.open_dataset(
            f, engine="h5netcdf", decode_cf=True, mask_and_scale=True,
            backend_kwargs={"phony_dims":"sort"}
        )
    except Exception:
        try:
            return xr.open_dataset(f, engine="scipy", decode_cf=True, mask_and_scale=True)
        except Exception:
            return xr.open_dataset(f, decode_cf=True, mask_and_scale=True)

def dataset_to_df_generic(ds, prefer_vars=None):
    """xr.Dataset -> DataFrame (mantém variáveis >0D; senão, linha escalar)."""
    ds = ds.reset_coords(drop=True)
    if prefer_vars:
        keep = [v for v in prefer_vars if v in ds.data_vars]
        if keep: ds = ds[keep]
    non_scalar = [v for v in ds.data_vars if ds[v].ndim > 0]
    if non_scalar:
        return ds[non_scalar].to_dataframe().reset_index()
    row = {}
    for v in ds.data_vars:
        val = ds[v].values
        try:
            row[v] = val.item() if getattr(val, "shape", ()) == () else np.asarray(val).ravel()[0].item()
        except Exception:
            row[v] = str(val)
    return pd.DataFrame([row]) if row else pd.DataFrame()

def collect_month_nc_l3m(in_dir: str, anos=(2002,2024), month=7):
    """Lista .nc do mês especificado no período, apenas em L3m, retornando DF com meta."""
    ncs = sorted(glob.glob(os.path.join(in_dir, "*.nc")))
    rows = []
    for f in ncs:
        dt = parse_date_from_name(f)
        ok = (dt is not None and anos[0] <= dt.year <= anos[1] and dt.month == month)
        rows.append({
            "folder": in_dir,
            "basename": os.path.basename(f),
            "fullpath": f,
            "dt": dt,
            "is_month_in_range": ok
        })
    df = pd.DataFrame(rows)
    return df[df["is_month_in_range"] == True].copy() if not df.empty else pd.DataFrame()

def check_missing_for_folder_l3m(in_dir: str):
    """Retorna DF de pendências do mês (sem csv/csv.gz) em L3m."""
    m_df = collect_month_nc_l3m(in_dir, anos=ANOS, month=MONTH)
    if m_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    out_dir = os.path.join(in_dir, "csv_export")
    os.makedirs(out_dir, exist_ok=True)
    miss_rows, ok_rows = [], []
    for _, row in m_df.iterrows():
        base = os.path.splitext(row["basename"])[0]
        gz = os.path.join(out_dir, base + ".csv.gz")
        plain = os.path.join(out_dir, base + ".csv")
        if os.path.isfile(gz) or os.path.isfile(plain):
            ok_rows.append(row.to_dict())
        else:
            miss_rows.append(row.to_dict())
    return pd.DataFrame(miss_rows), pd.DataFrame(ok_rows)

def apply_bbox_df(df: pd.DataFrame, bbox):
    """Aplica BBOX quando colunas de lat/lon existirem."""
    if bbox is None:
        return df, False
    lon_min, lon_max, lat_min, lat_max = bbox
    lon_cols = [c for c in ["lon","longitude","x"] if c in df.columns]
    lat_cols = [c for c in ["lat","latitude","y"] if c in df.columns]
    if not lon_cols or not lat_cols:
        return df, False
    lon_c, lat_c = lon_cols[0], lat_cols[0]
    mask = (df[lon_c] >= lon_min) & (df[lon_c] <= lon_max) & \
           (df[lat_c] >= lat_min) & (df[lat_c] <= lat_max)
    return df.loc[mask].copy(), True

print("   OK utilitários prontos.")

# ------------------------- DETECÇÃO DE PENDÊNCIAS -------------------------
print(f"\n▶ Passo 2/6: Checando pendências do mês {MONTH} (2002–2024) em L3m...")
if not os.path.isdir(L3M_DIR):
    raise SystemExit(f"⛔ Pasta L3m não encontrada: {L3M_DIR}. Certifique-se de que a estrutura 'Dados/DAY/L3m' existe neste repositório.")

miss_df, ok_df = check_missing_for_folder_l3m(L3M_DIR)
print(f"   - Arquivos no mês no período: {len(miss_df) + len(ok_df)}")
print(f"   - Já convertidos: {len(ok_df)} | Pendentes: {len(miss_df)}")

# ------------------------- CONVERTER PENDÊNCIAS (L3m) -------------------------
print(f"\n▶ Passo 3/6: Convertendo SOMENTE pendências do mês {MONTH} em L3m (com BBOX)...")
conv_ok = 0
conv_fail = 0
converted_paths = []
bbox_used_count = 0

if miss_df.empty:
    print("   ✓ Não há pendências em L3m. Nada a converter.")
else:
    for _, row in miss_df.iterrows():
        f = row["fullpath"]
        base = os.path.splitext(os.path.basename(f))[0]
        out_dir = ensure_dir(os.path.join(L3M_DIR, "csv_export"))
        out_ext = ".csv.gz" if COMPRESS_GZ else ".csv"
        out_path = os.path.join(out_dir, base + out_ext)
        print(f"   → Convertendo: {os.path.basename(f)} (L3m)")

        try:
            ds = open_ds_any(f)
            dt = extract_datetime(ds, f)
            # Segurança: reconfirmar mês/período
            if dt is None or not (ANOS[0] <= dt.year <= ANOS[1]) or dt.month != MONTH:
                print("     [skip] Data não pertence ao mês/período configurado.")
                ds.close()
                continue

            # Converte para DataFrame (grade regular)
            df = dataset_to_df_generic(ds, prefer_vars=VARS_TO_KEEP)

            # Reorganiza: coords primeiro
            lead_coords = [c for c in ["lat","latitude","y","i","j"] if c in df.columns] + \
                          [c for c in ["lon","longitude","x","row","col","bin"] if c in df.columns]
            data_cols = [c for c in df.columns if c not in lead_coords]
            if data_cols:
                df = df[lead_coords + data_cols]

            # Aplica BBOX (se possível)
            df, used = apply_bbox_df(df, BBOX)
            bbox_used_count += int(used)
            if BBOX is not None and not used:
                print("     [aviso] BBOX definido, mas não aplicado (colunas lat/lon não detectadas).")

            # carimbo temporal detalhado
            Y, M, D = dt.year, dt.month, dt.day
            h, mi, s = dt.hour, dt.minute, dt.second
            df.insert(0, "year",   Y)
            df.insert(1, "month",  M)
            df.insert(2, "day",    D)
            df.insert(3, "hour",   h)
            df.insert(4, "minute", mi)
            df.insert(5, "second", s)
            df.insert(6, "date", f"{Y:04d}-{M:02d}-{D:02d}")
            df.insert(7, "datetime_iso", f"{Y:04d}-{M:02d}-{D:02d}T{h:02d}:{mi:02d}:{s:02d}")

            # ordenar por tempo (quando fizer sentido)
            try:
                df = df.sort_values(["year","month","day","hour","minute","second"])
            except Exception:
                pass

            # salvar
            if COMPRESS_GZ:
                df.to_csv(out_path, index=False, compression="infer")
            else:
                df.to_csv(out_path, index=False)

            converted_paths.append(out_path)
            conv_ok += 1
            print(f"     ✓ salvo: {out_path} (linhas: {len(df)})")
            ds.close()

        except Exception as e:
            conv_fail += 1
            print(f"     [ERRO] {os.path.basename(f)} -> {e}")
            traceback.print_exc(limit=1)

# ------------------------- REVALIDAÇÃO -------------------------
print(f"\n▶ Passo 4/6: Revalidando conversões do mês {MONTH} em L3m...")
miss2_df, ok2_df = check_missing_for_folder_l3m(L3M_DIR)
print(f"   Pendências restantes: {len(miss2_df)}")

# ------------------------- RESUMO -------------------------
print("\n▶ Passo 5/6: Resumo (L3m)")
print(f"  Convertidos agora          : {conv_ok}")
print(f"  Falhas na conversão        : {conv_fail}")
print(f"  Pendências finais (L3m)    : {len(miss2_df)}")
if BBOX is not None:
    print(f"  BBOX aplicado (arquivos)   : {bbox_used_count}")

# ------------------------- LISTAR NOVOS/ PENDÊNCIAS -------------------------
print("\n▶ Passo 6/6: Listando novos arquivos gerados e pendências finais (se houver)...")
if converted_paths:
    print("  Arquivos gerados (amostra):")
    for p in converted_paths[:20]:
        print("   -", p)
else:
    print("  (Nenhum novo arquivo criado nesta execução.)")

# opcional: salvar CSV com pendências restantes
if len(miss2_df) > 0:
    out_missing = os.path.join(L3M_DIR, "csv_export", f"pendencias_mes_{MONTH:02d}_sem_csv_restantes_L3m.csv")
    try:
        os.makedirs(os.path.dirname(out_missing), exist_ok=True)
        miss2_df.to_csv(out_missing, index=False)
        print("  ✓ Pendências restantes salvas em:", out_missing)
    except Exception as e:
        print("  ⚠ Falha ao salvar pendências restantes em", out_missing, "->", e)

print("\n✅ FIM (L3m, mês =", MONTH, ")")