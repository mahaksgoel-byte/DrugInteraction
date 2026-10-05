"""Write a checksum-backed inventory of files present under data/raw."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
OUT = RAW / "DATA_PROVENANCE.yaml"
TODAY = date(2026, 10, 5).isoformat()


def source_for(path: Path) -> dict[str, str]:
    name = path.name
    rel = path.relative_to(RAW).as_posix()
    if rel.startswith("ddinter/"):
        if name == "drug_data.json":
            return {
                "source": "DDInter metadata crosswalk (supplemental Biothings scraper export)",
                "version": "repository main snapshot retrieved 2026-10-05; underlying DDInter data",
                "url": "https://raw.githubusercontent.com/mnarayan1/DDInter/main/drug_data.json",
            }
        return {
            "source": "DDInter 2.0",
            "version": "downloaded 2026-10-05",
            "url": f"https://ddinter2.scbdd.com/static/media/download/{name}",
        }
    if rel.startswith("sider/"):
        return {
            "source": "SIDER",
            "version": "4.1 (2015-10-21)",
            "url": f"https://sideeffects.embl.de/media/download/{name}",
        }
    if rel.startswith("pharmgkb/"):
        endpoint = name
        version = "ClinPGx export retrieved 2026-10-05" if name in {"chemicals.zip", "drugLabels.zip", "genes.zip"} else "2025-07-05 export"
        return {
            "source": "ClinPGx (formerly PharmGKB)",
            "version": version,
            "url": f"https://api.clinpgx.org/v1/download/file/data/{endpoint}",
        }
    if rel.startswith("pubchem/"):
        return {
            "source": "PubChem PUG REST",
            "version": "API response retrieved 2026-10-05",
            "url": "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/{comma-separated-CIDs}/property/Title,CanonicalSMILES,IsomericSMILES,InChIKey/JSON",
        }
    if rel.startswith("string/"):
        string_table = name.removeprefix("9606.").removesuffix(".txt.gz")
        return {
            "source": "STRING",
            "version": "12.5; Homo sapiens (taxon 9606)",
            "url": f"https://stringdb-downloads.org/download/{string_table}/" + name,
        }
    if rel.startswith("uniprot/"):
        return {
            "source": "UniProt reference proteomes",
            "version": "release 2026-09-15; UP000005640, Homo sapiens",
            "url": "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/reference_proteomes/Eukaryota/UP000005640/UP000005640_9606.fasta.gz",
        }
    if rel.startswith("chembl/"):
        return {
            "source": "ChEMBL",
            "version": "37; official molecule representations release downloaded 2026-10-05",
            "url": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/chembl_37_chemreps.txt.gz",
        }
    if rel.startswith("unichem/"):
        if name == "source.tsv.gz":
            url = "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/table_dumps/source.tsv.gz"
            version = "UniChem mapping snapshot 2026-10-03"
        elif name == "src1src22.txt.gz":
            url = "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id1/src1src22.txt.gz"
            version = "UniChem ChEMBL (source 1) to PubChem (source 22), snapshot 2026-10-03"
        else:
            url = "https://ftp.ebi.ac.uk/pub/databases/chembl/UniChem/data/wholeSourceMapping/src_id2/src2src22.txt.gz"
            version = "UniChem DrugBank (source 2) to PubChem (source 22), snapshot 2026-10-03"
        return {"source": "UniChem", "version": version, "url": url}
    return {"source": "unknown", "version": "unknown", "url": ""}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    files = []
    for path in sorted(p for p in RAW.rglob("*") if p.is_file() and p != OUT):
        entry = source_for(path)
        files.append(
            {
                "local_path": path.relative_to(ROOT).as_posix(),
                "source": entry["source"],
                "version": entry["version"],
                "downloaded_on": TODAY,
                "url": entry["url"],
                "size_bytes": path.stat().st_size,
                "sha256": digest(path),
            }
        )
    manifest = {
        "schema_version": 1,
        "generated_on": TODAY,
        "access_notes": [
            "DrugBank interaction/target/enzyme tables are not present: download requires the account and academic licence described by plan.md.",
            "UniChem DrugBank-to-PubChem cross-references are present and are not a substitute for the gated DrugBank interaction dataset.",
            "PubChem JSON responses are scoped to DDInter cross-reference CIDs, not a whole-database dump.",
            "DDInter drug_data.json is a supplemental third-party crosswalk export; its provenance is recorded separately from official DDInter category files.",
            "ClinPGx archives include their own LICENSE.txt files; review those terms before redistribution.",
        ],
        "files": files,
    }
    # JSON is valid YAML 1.2 and avoids an extra YAML package dependency.
    OUT.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(files)} file checksums to {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
