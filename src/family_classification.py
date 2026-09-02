"""Protein family classification (kinase / GPCR / protease).

Primary source is curator-annotated UniProt keywords; a curated keyword
fallback is applied only to proteins without UniProt annotation. A protein
matching more than one family is flagged ambiguous and excluded.
"""
import re
import time

import pandas as pd
import requests


UNIPROT_ACCESSION_RE = re.compile(
    r"^(?:[OPQ][0-9][A-Z0-9]{3}[0-9]|"
    r"[A-NR-Z][0-9](?:[A-Z][A-Z0-9]{2}[0-9]){1,2})(?:-[0-9]+)?$"
)


def normalize_uniprot_accession(value):
    """Return a normalized UniProt accession, or None for malformed values."""
    if pd.isna(value):
        return None
    accession = str(value).strip().upper()
    return accession if UNIPROT_ACCESSION_RE.fullmatch(accession) else None


def _entry_to_annotation(entry):
    kws = [kw.get("name", "") for kw in entry.get("keywords", [])]

    fams = []
    for comment in entry.get("comments", []):
        if comment.get("commentType") == "SIMILARITY":
            fams.extend(
                text.get("value", "")
                for text in comment.get("texts", [])
                if text.get("value")
            )

    recommended = (
        entry.get("proteinDescription", {}).get("recommendedName", {})
    )
    protein_name = (
        recommended.get("fullName", {}).get("value", "")
        if recommended else ""
    )
    return {
        "keywords": kws,
        "protein_families": fams,
        "recommended_name": protein_name,
    }


def fetch_uniprot_annotation_batch(
    uniprot_ids,
    batch_size=25,
    sleep=0.3,
    max_retries=4,
    return_report=False,
):
    """Batch-query keyword + family info from the UniProt REST API.

    Returns {uid: {'keywords': [...], 'protein_families': [...],
    'recommended_name': str}}.
    """
    results = {}
    raw_ids = sorted({str(uid).strip().upper() for uid in uniprot_ids
                      if not pd.isna(uid)})
    valid_ids = [uid for uid in raw_ids
                 if normalize_uniprot_accession(uid) is not None]
    invalid_ids = sorted(set(raw_ids) - set(valid_ids))
    total = len(raw_ids)

    pending = [valid_ids[i:i + batch_size]
               for i in range(0, len(valid_ids), batch_size)]
    n_requests = 0
    request_failures = []
    not_returned = set()
    headers = {"User-Agent": "plba-generalization/1.0 (UniProt annotation)"}

    if invalid_ids:
        print(f"  [WARN] skipped {len(invalid_ids):,} malformed UniProt ID(s)")

    while pending:
        batch = pending.pop(0)
        n_requests += 1
        query = " OR ".join(f"accession:{uid}" for uid in batch)
        params = {
            "query": query,
            "fields": "accession,keyword,protein_name,cc_similarity",
            "format": "json",
            "size": batch_size,
        }

        response = None
        try:
            for attempt in range(max_retries + 1):
                response = requests.get(
                    "https://rest.uniprot.org/uniprotkb/search",
                    params=params,
                    headers=headers,
                    timeout=60,
                )
                if response.status_code == 200:
                    break
                if response.status_code == 429 or response.status_code >= 500:
                    if attempt < max_retries:
                        retry_after = response.headers.get("Retry-After")
                        delay = (float(retry_after) if retry_after else
                                 sleep * (2 ** attempt))
                        print(f"  [WARN] HTTP {response.status_code}; "
                              f"retrying in {delay:.1f}s")
                        time.sleep(delay)
                        continue
                break

            if response is None or response.status_code != 200:
                status = response.status_code if response is not None else None
                # UniProt may reject a long OR query. Split it until the exact
                # offending accession (if any) is isolated.
                if status == 400 and len(batch) > 1:
                    midpoint = len(batch) // 2
                    pending[0:0] = [batch[:midpoint], batch[midpoint:]]
                    print(f"  [WARN] HTTP 400 for {len(batch)} accessions; "
                          "retrying as smaller batches")
                    time.sleep(sleep)
                    continue
                detail = (response.text.replace("\n", " ")[:200]
                          if response is not None else "no response")
                print(f"  [WARN] request {n_requests}: HTTP {status} "
                      f"for {batch!r} | {detail}")
                request_failures.append({
                    "accessions": batch,
                    "status": status,
                    "detail": detail,
                })
                time.sleep(sleep * 3)
                continue

            entries = response.json().get("results", [])
            matched_ids = set()
            for entry in entries:
                primary = str(entry.get("primaryAccession", "")).upper()
                if not primary:
                    continue
                aliases = {primary}
                aliases.update(str(x).upper()
                               for x in entry.get("secondaryAccessions", []))
                annotation = _entry_to_annotation(entry)

                # Cache under the requested identifier. This is important for
                # obsolete secondary accessions and isoform identifiers whose
                # API result is returned under a canonical primary accession.
                requested_matches = [
                    uid for uid in batch
                    if uid in aliases or uid.split("-", 1)[0] in aliases
                ]
                if len(batch) == 1 and not requested_matches:
                    requested_matches = batch
                for uid in requested_matches:
                    results[uid] = annotation
                    matched_ids.add(uid)

            not_returned.update(set(batch) - matched_ids)
            if n_requests % 10 == 0:
                print(f"  Progress: requests={n_requests:,} | "
                      f"retrieved={len(results):,}, "
                      f"terminal_failures={len(request_failures):,}, "
                      f"pending_batches={len(pending):,}")

        except Exception as e:
            print(f"  [ERROR] request {n_requests}: {e}")
            request_failures.append({
                "accessions": batch,
                "status": None,
                "detail": str(e)[:200],
            })
            time.sleep(sleep * 5)

        time.sleep(sleep)

    terminal_failed_ids = sorted({
        uid for failure in request_failures for uid in failure["accessions"]
    })
    not_returned.difference_update(results)
    not_returned.difference_update(terminal_failed_ids)
    report = {
        "requested": total,
        "valid_requested": len(valid_ids),
        "retrieved": len(results),
        "invalid_ids": invalid_ids,
        "terminal_failed_ids": terminal_failed_ids,
        "not_returned_ids": sorted(not_returned),
        "request_failures": request_failures,
    }
    print(f"\nFinal: {len(results):,} retrieved, "
          f"{len(terminal_failed_ids):,} request-failed, "
          f"{len(not_returned):,} not found, "
          f"{len(invalid_ids):,} malformed")
    return (results, report) if return_report else results


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
