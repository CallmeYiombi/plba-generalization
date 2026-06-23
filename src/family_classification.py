"""Protein family classification (kinase / GPCR / protease).

Primary source is curator-annotated UniProt keywords; a curated keyword
fallback is applied only to proteins without UniProt annotation. A protein
matching more than one family is flagged ambiguous and excluded.
"""
import re
import time

import pandas as pd
import requests


def fetch_uniprot_annotation_batch(uniprot_ids, batch_size=100, sleep=0.3):
    """Batch-query keyword + family info from the UniProt REST API.

    Returns {uid: {'keywords': [...], 'protein_families': [...],
    'recommended_name': str}}.
    """
    results = {}
    uniprot_ids = sorted(set(uniprot_ids))
    total = len(uniprot_ids)
    n_success = n_fail = 0

    for i in range(0, total, batch_size):
        batch = uniprot_ids[i:i + batch_size]
        query = " OR ".join(f"accession:{uid}" for uid in batch)
        params = {
            "query": query,
            "fields": "accession,keyword,protein_name,cc_similarity",
            "format": "json",
            "size": batch_size,
        }

        try:
            r = requests.get("https://rest.uniprot.org/uniprotkb/search",
                             params=params, timeout=60)
            if r.status_code != 200:
                print(f"  [WARN] batch {i // batch_size + 1}: HTTP {r.status_code}")
                n_fail += len(batch)
                time.sleep(sleep * 3)
                continue

            entries = r.json().get("results", [])
            for entry in entries:
                acc = entry.get("primaryAccession")
                if not acc:
                    continue

                kws = [kw.get("name", "") for kw in entry.get("keywords", [])]

                fams = []
                for cm in entry.get("comments", []):
                    if cm.get("commentType") == "SIMILARITY":
                        fams += [t.get("value", "") for t in cm.get("texts", [])
                                 if t.get("value")]

                rec = entry.get("proteinDescription", {}).get("recommendedName", {})
                pname = rec.get("fullName", {}).get("value", "") if rec else ""

                results[acc] = {
                    "keywords": kws,
                    "protein_families": fams,
                    "recommended_name": pname,
                }

            n_success += len(entries)
            if (i // batch_size + 1) % 10 == 0:
                pct = min(100.0, (i + batch_size) / total * 100)
                print(f"  Progress: {i + batch_size:,}/{total:,} ({pct:.1f}%) | "
                      f"success={n_success:,}, fail={n_fail:,}")

        except Exception as e:
            print(f"  [ERROR] batch {i // batch_size + 1}: {e}")
            n_fail += len(batch)
            time.sleep(sleep * 5)

        time.sleep(sleep)

    print(f"\nFinal: {n_success:,} retrieved, {n_fail:,} failed, "
          f"{total - n_success - n_fail:,} not in response")
    return results


# UniProt keyword -> family mapping (https://www.uniprot.org/keywords/)
UNIPROT_KW_FAMILY = {
    "kinase": [
        "Kinase",
        "Tyrosine-protein kinase",
        "Serine/threonine-protein kinase",
    ],
    "gpcr": [
        "G-protein coupled receptor",
    ],
    "protease": [
        "Protease",
        "Serine protease",
        "Cysteine protease",
        "Aspartyl protease",
        "Metalloprotease",
        "Thiol protease",
    ],
}


def classify_by_uniprot(uid, uniprot_anno):
    """Return 'kinase'/'gpcr'/'protease', 'ambiguous_*', or None."""
    info = uniprot_anno.get(uid)
    if info is None:
        return None

    kws = set(info.get("keywords", []))
    matched = [fam for fam, fam_kws in UNIPROT_KW_FAMILY.items()
               if any(k in kws for k in fam_kws)]

    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]
    return "ambiguous_" + "_".join(sorted(matched))


KINASE_KEYWORDS = [
    "kinase", "protein kinase",
    "tyrosine kinase", "serine/threonine kinase",
    "serine/threonine-protein kinase",
    "cyclin-dependent kinase",
    "mitogen-activated protein kinase", "map kinase",
    "phosphoinositide 3-kinase", "phosphatidylinositol 3-kinase",
    "janus kinase", "casein kinase",
    "aurora kinase", "polo-like kinase",
    "rho-associated protein kinase",
    "raf kinase", "src kinase", "abl kinase",
]

# True GPCR keywords only; standalone 'receptor' is deliberately excluded.
GPCR_INCLUDE = [
    "g protein-coupled receptor", "g-protein-coupled receptor",
    "g protein coupled receptor", "g-protein coupled receptor", "gpcr",
    "adrenergic receptor", "dopamine receptor",
    "serotonin receptor", "5-hydroxytryptamine receptor",
    "muscarinic acetylcholine receptor", "muscarinic receptor",
    "opioid receptor", "histamine receptor",
    "cannabinoid receptor", "chemokine receptor",
    "adenosine receptor", "purinergic receptor",
    "neuropeptide receptor", "somatostatin receptor",
    "tachykinin receptor", "vasopressin receptor",
    "oxytocin receptor", "melatonin receptor",
    "prostanoid receptor", "prostaglandin receptor",
    "leukotriene receptor", "lysophosphatidic acid receptor",
    "sphingosine 1-phosphate receptor",
    "free fatty acid receptor",
    "metabotropic glutamate receptor",
    "olfactory receptor", "taste receptor",
    "gonadotropin-releasing hormone receptor",
    "thyrotropin-releasing hormone receptor",
    "corticotropin-releasing factor receptor",
    "bradykinin receptor", "angiotensin receptor",
    "endothelin receptor", "calcitonin receptor",
    "glucagon receptor", "secretin receptor",
]

# Non-GPCR receptors to remove (RTKs, nuclear, immune/cytokine, ion channels).
GPCR_EXCLUDE = [
    "receptor tyrosine kinase", "tyrosine-protein kinase receptor",
    "insulin receptor", "insulin-like growth factor",
    "epidermal growth factor receptor", "egfr",
    "fibroblast growth factor receptor", "fgfr",
    "vascular endothelial growth factor receptor", "vegfr",
    "platelet-derived growth factor receptor", "pdgfr",
    "hepatocyte growth factor receptor",
    "nerve growth factor receptor",
    "nuclear receptor", "estrogen receptor", "androgen receptor",
    "glucocorticoid receptor", "mineralocorticoid receptor",
    "progesterone receptor", "retinoic acid receptor",
    "retinoid x receptor", "thyroid hormone receptor",
    "vitamin d receptor", "peroxisome proliferator",
    "liver x receptor", "farnesoid x receptor",
    "constitutive androstane receptor", "pregnane x receptor",
    "toll-like receptor", "toll like receptor",
    "interleukin receptor", "interleukin-",
    "tumor necrosis factor receptor", "tnf receptor",
    "t-cell receptor", "t cell receptor",
    "b-cell receptor", "b cell receptor",
    "fc receptor", "death receptor",
    "transient receptor potential", "trp channel",
    "ion channel", "ligand-gated ion channel",
    "nicotinic acetylcholine receptor", "gaba receptor",
    "ionotropic glutamate receptor", "glutamate receptor ionotropic",
    "nmda receptor", "ampa receptor", "kainate receptor",
    "glycine receptor",
]

PROTEASE_KEYWORDS = [
    "protease", "peptidase", "proteinase",
    "endopeptidase", "exopeptidase",
    "serine protease", "cysteine protease",
    "aspartic protease", "aspartyl protease",
    "metalloprotease", "metalloproteinase",
    "caspase", "cathepsin",
    "thrombin", "trypsin", "chymotrypsin",
    "elastase", "kallikrein", "renin",
    "matrix metalloproteinase",
]


def normalize_text(x):
    if pd.isna(x):
        return ""
    x = str(x).lower().replace("_", " ").replace("-", " ")
    return re.sub(r"\s+", " ", x).strip()


def keyword_match(text, keywords):
    text = normalize_text(text)
    for kw in keywords:
        if re.search(r"\b" + re.escape(normalize_text(kw)) + r"\b", text):
            return True
    return False


def classify_by_keyword(protein_name, target_name):
    """Keyword fallback; used only when UniProt annotation is missing."""
    text = f"{normalize_text(protein_name)} {normalize_text(target_name)}"

    is_kinase = keyword_match(text, KINASE_KEYWORDS)
    is_gpcr = keyword_match(text, GPCR_INCLUDE) and not keyword_match(text, GPCR_EXCLUDE)
    is_protease = keyword_match(text, PROTEASE_KEYWORDS)

    matched = [name for name, hit in
               (("kinase", is_kinase), ("gpcr", is_gpcr), ("protease", is_protease))
               if hit]

    if not matched:
        return None
    if len(matched) == 1:
        return matched[0]
    return "ambiguous_" + "_".join(matched)


def classify_with_source(row, uniprot_anno):
    """Return (family, source) where source is 'uniprot', 'keyword', or 'none'."""
    uniprot_result = classify_by_uniprot(row["uniprot_id"], uniprot_anno)
    if uniprot_result is not None:
        return pd.Series([uniprot_result, "uniprot"])
    kw_result = classify_by_keyword(row.get("protein_name"), row.get("target_name"))
    if kw_result is not None:
        return pd.Series([kw_result, "keyword"])
    return pd.Series([None, "none"])
