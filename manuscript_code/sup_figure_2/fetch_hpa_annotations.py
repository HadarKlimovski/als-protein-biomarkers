"""Fetch Human Protein Atlas per-gene annotations for the 9 discovery proteins
   (TLV adjusted univariate, overlap-FDR < 0.05) and write hpa_protein_summary.csv.

   Reproduces the schema of the original CSV; dict/list fields are serialized as
   Python-literal strings for readability in Excel.

   Run from this directory:   python hpa_fetch.py
   Requires outbound HTTPS to mygene.info + proteinatlas.org; pip deps: requests, pandas.
"""

import json
import time
from pathlib import Path
import requests
import pandas as pd

PROTEINS = ['AFM', 'CRTAC1', 'GRN', 'GSN', 'IGFBP2', 'IL1RAP', 'MEGF10', 'PEBP4', 'SFTPD']

OUT_CSV = Path(__file__).with_name('hpa_protein_summary.csv')

# --- Gene symbol -> Ensembl gene ID ------------------------------------------

def symbol_to_ensg(symbol):
    url = f'https://mygene.info/v3/query?q=symbol:{symbol}&species=human&fields=ensembl.gene'
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    for hit in r.json().get('hits', []):
        ens = hit.get('ensembl')
        if isinstance(ens, dict) and ens.get('gene'):
            return ens['gene']
        if isinstance(ens, list):
            for e in ens:
                if e.get('gene'):
                    return e['gene']
    raise RuntimeError(f'No Ensembl gene id for symbol {symbol}')

# --- HPA JSON fetch ----------------------------------------------------------

def fetch_hpa(ensg):
    r = requests.get(f'https://www.proteinatlas.org/{ensg}.json', timeout=30)
    r.raise_for_status()
    return r.json()

# HPA field names change over time; list candidates in order of preference.
FIELD_MAP = {
    'gene_description':     ['Gene description'],
    'protein_class':        ['Protein class'],
    'biological_process':   ['Biological process'],
    'disease_involvement':  ['Disease involvement'],
    'tissue_specificity':   ['RNA tissue specificity'],
    'tissue_specific_nTPM': ['RNA tissue specific nTPM', 'RNA tissue specific NX'],
    'single_cell_specificity': ['RNA single cell type specificity'],
    'single_cell_specific':    ['RNA single cell type specific nTPM'],
    'brain_specificity':    ['RNA brain regional specificity'],
    'brain_specific':       ['RNA brain regional specific nTPM', 'RNA brain regional specific NX'],
    'blood_specificity':    ['RNA blood cell specificity'],
    'blood_specific':       ['RNA blood cell specific nTPM', 'RNA blood cell specific NX'],
    'tissue_cell_enrichment': ['Tissue cell type enrichment', 'RNA tissue cell type enrichment'],
    'tissue_cluster':       ['RNA tissue cluster', 'Tissue expression cluster'],
    'brain_cluster':        ['RNA brain regional cluster'],
    'single_cell_cluster':  ['RNA single cell type cluster'],
}

def extract(d, symbol):
    out = {'protein': symbol}
    for col, candidates in FIELD_MAP.items():
        v = None
        for k in candidates:
            if k in d:
                v = d[k]; break
        if isinstance(v, (list, dict)):
            v = str(v)
        out[col] = v
    return out

def main():
    rows = []
    for p in PROTEINS:
        try:
            ensg = symbol_to_ensg(p)
            hpa = fetch_hpa(ensg)
            row = extract(hpa, p)
            rows.append(row)
            print(f'OK  {p:8s} -> {ensg}')
        except Exception as e:
            print(f'ERR {p}: {e}')
            rows.append({'protein': p})
        time.sleep(0.3)  # be polite to the public API
    cols = ['protein'] + list(FIELD_MAP.keys())
    pd.DataFrame(rows)[cols].to_csv(OUT_CSV, index=False)
    print(f'\nWrote {OUT_CSV} ({len(rows)} rows)')

if __name__ == '__main__':
    main()
