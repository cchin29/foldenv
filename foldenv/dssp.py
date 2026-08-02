"""DSSP → secondary structure (H/E/C) + RSA.

Runs mkdssp (via Biopython's `dssp_dict_from_pdb_file`, which auto-detects the version and
passes `--output-format=dssp` for v4 — the D6 mmCIF gotcha), then:
  * maps the 8-state SS down to 3 (`H/G/I → H`, `E/B → E`, else `C`);
  * normalizes DSSP's **absolute** ASA to RSA with an explicit MaxASA table (D3), so the
    `rsa.max_asa_table` config choice is honored and recorded rather than hidden inside
    Biopython's single built-in table.

Using the absolute ASA (not Biopython's pre-normalized relative ASA) is deliberate: it lets
us switch MaxASA references without re-running DSSP and keeps the normalization auditable.
"""
from __future__ import annotations

import re
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

SS8_TO_SS3 = {
    "H": "H", "G": "H", "I": "H",   # α / 3-10 / π helix
    "E": "E", "B": "E",             # β-strand / β-bridge
    "T": "C", "S": "C", "-": "C", "P": "C", " ": "C",  # turn / bend / PPII / coil
}

# --- MaxASA reference tables (Å²), keyed by 1-letter AA -------------------------------
# Tien et al. 2013, PLoS ONE 9(11):e80635, Table 1 (theoretical & empirical); Sander & Rost
# 1994. Record which was used — absolute RSA shifts ~0.05–0.1 between tables.

_TIEN2013_THEORETICAL = {
    "A": 129.0, "R": 274.0, "N": 195.0, "D": 193.0, "C": 167.0, "E": 223.0, "Q": 225.0,
    "G": 104.0, "H": 224.0, "I": 197.0, "L": 201.0, "K": 236.0, "M": 224.0, "F": 240.0,
    "P": 159.0, "S": 155.0, "T": 172.0, "W": 285.0, "Y": 263.0, "V": 174.0,
}
_TIEN2013_EMPIRICAL = {
    "A": 121.0, "R": 265.0, "N": 187.0, "D": 187.0, "C": 148.0, "E": 214.0, "Q": 214.0,
    "G": 97.0, "H": 216.0, "I": 195.0, "L": 191.0, "K": 230.0, "M": 203.0, "F": 228.0,
    "P": 154.0, "S": 143.0, "T": 163.0, "W": 264.0, "Y": 255.0, "V": 165.0,
}
_SANDER_ROST1994 = {
    "A": 106.0, "R": 248.0, "N": 157.0, "D": 163.0, "C": 135.0, "E": 194.0, "Q": 198.0,
    "G": 84.0, "H": 184.0, "I": 169.0, "L": 164.0, "K": 205.0, "M": 188.0, "F": 197.0,
    "P": 136.0, "S": 130.0, "T": 142.0, "W": 227.0, "Y": 222.0, "V": 142.0,
}
MAX_ASA_TABLES = {
    "tien2013_theoretical": _TIEN2013_THEORETICAL,
    "tien2013_empirical": _TIEN2013_EMPIRICAL,
    "sander_rost1994": _SANDER_ROST1994,
}


@dataclass
class ResidueDSSP:
    resnum: int          # author residue number (AF: canonical UniProt numbering)
    aa: str              # 1-letter amino acid from DSSP
    ss3: str             # H / E / C
    ss8: str             # raw DSSP 8-state
    acc: float           # absolute ASA (Å²)
    rsa: float           # acc / MaxASA(aa), clamped to [0, 1]


def _detect_version(exe: str) -> str:
    """Parse `mkdssp --version` → semantic version string (default 4.x if unparseable)."""
    try:
        out = subprocess.check_output([exe, "--version"], text=True, stderr=subprocess.STDOUT)
        m = re.search(r"(\d+\.\d+\.\d+)", out) or re.search(r"(\d+\.\d+)", out)
        if m:
            v = m.group(1)
            return v if v.count(".") == 2 else v + ".0"
    except (OSError, subprocess.SubprocessError):
        pass
    # assume modern mkdssp; Biopython then passes --output-format=dssp. Warn, because a
    # real mkdssp 3.x whose banner didn't parse would be driven with a v4-only flag.
    warnings.warn(
        f"Could not parse `{exe} --version`; assuming mkdssp 4.x (--output-format=dssp). "
        "If this is mkdssp 3.x, set dssp.executable explicitly or upgrade.",
        RuntimeWarning, stacklevel=2,
    )
    return "4.0.0"


def max_asa(aa: str, table: str) -> float | None:
    """MaxASA for a 1-letter AA under `table`; None for non-standard residues (→ RSA NaN).

    Expects a canonical uppercase code — `run_dssp` normalizes DSSP's lowercase disulfide
    cysteines to 'C' first. No `.upper()` here on purpose: a stray lowercase letter should
    miss (→ None → NaN) loudly rather than silently resolve to the wrong residue.
    """
    return MAX_ASA_TABLES[table].get(aa)


def run_dssp(
    cif_path: str | Path,
    *,
    chain_id: str | None = None,
    exe: str = "mkdssp",
    max_asa_table: str = "tien2013_theoretical",
) -> dict[int, ResidueDSSP]:
    """Run DSSP on `cif_path`; return {resnum: ResidueDSSP} for one chain.

    Args:
        chain_id: chain to keep; None → the first chain seen (AF monomers are one chain).
        exe: DSSP executable (D6 config `dssp.executable`).
        max_asa_table: D3 MaxASA reference (key of MAX_ASA_TABLES).
    """
    if max_asa_table not in MAX_ASA_TABLES:
        raise ValueError(
            f"Unknown max_asa_table {max_asa_table!r}; choose {sorted(MAX_ASA_TABLES)}"
        )
    from Bio.PDB.DSSP import dssp_dict_from_pdb_file

    version = _detect_version(exe)
    dssp_dict, keys = dssp_dict_from_pdb_file(str(cif_path), DSSP=exe, dssp_version=version)

    # An empty result is never legitimate for a real structure — returning {} here would
    # give every residue rsa=None/ss=None downstream (a fully null-structural protein) with
    # no error. Fail instead (usually a mis-invoked/failed mkdssp).
    if not keys:
        raise RuntimeError(
            f"DSSP ({exe}) produced no residues for {cif_path} — empty output. "
            "Check the mkdssp install / mmCIF; a null-structural profile would poison RSA/SS."
        )
    if chain_id is None:
        chain_id = keys[0][0]

    out: dict[int, ResidueDSSP] = {}
    for key in keys:
        ch, res_id = key
        if ch != chain_id:
            continue
        _het, resnum, _icode = res_id
        aa, ss, acc = dssp_dict[key][0], dssp_dict[key][1], dssp_dict[key][2]
        # DSSP renames disulfide-bonded cysteines to lowercase letters (a,b,c,… paired per
        # bridge). The low-level dssp_dict_from_pdb_file leaves them lowercase (only
        # Bio.PDB.DSSP's class restores them), so normalize back to 'C' — otherwise the
        # residue identity and its MaxASA lookup are wrong (e.g. 'a'→alanine).
        if aa.islower():
            aa = "C"
        ss8 = ss if ss and ss.strip() else "-"
        ss3 = SS8_TO_SS3.get(ss8, "C")
        m = max_asa(aa, max_asa_table)
        # acc and m are both ≥ 0, so only the upper clamp can bind.
        rsa = float("nan") if not m else min(acc / m, 1.0)
        out[int(resnum)] = ResidueDSSP(
            resnum=int(resnum), aa=aa, ss3=ss3, ss8=ss8, acc=float(acc), rsa=rsa
        )
    if not out:
        raise RuntimeError(
            f"DSSP ({exe}) returned residues but none on chain {chain_id!r} for {cif_path} "
            f"(chains present: {sorted({k[0] for k in keys})})."
        )
    return out
