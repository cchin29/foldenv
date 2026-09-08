"""DSSP → secondary structure (H/E/C) + RSA.

Runs mkdssp via Biopython's `dssp_dict_from_pdb_file`. That function does not probe the
executable itself (only Biopython's higher-level `DSSP` class does), so `_detect_version` reads
the version here and passes it in, which is what makes Biopython emit `--output-format=dssp`
for v4 — the D6 mmCIF gotcha. Then:
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

# DSSP's 8-state code reduced to 3 states. **There is more than one convention for this and they
# disagree**, so the choice is recorded here rather than left implicit. This is Cuff & Barton's
# **Method A** (1999, Proteins 34:508-519): G (3-10) and I (π) → H, B (isolated β-bridge) → E.
# MDTraj's `compute_dssp(simplified=True)` and MDAnalysis use the same mapping, so structural
# analysis tooling generally agrees with this file.
#
# The alternative is their **Method B**, which sends G, I and B → C. It is the convention of the
# secondary-structure-*prediction* literature (JPred/CASP-style Q3 scoring) rather than of
# structure-analysis tools, and it can only report less H and E, never more, since Method B's
# H- and E-sets are proper subsets of Method A's. The two genuinely disagree on real proteins:
# ~1% of TP53's residues flip between them and ~5.6% of TEM-1's. So comparing this field against
# a published secondary-structure figure is only meaningful once both reductions match.
#
# P (PPII / κ-helix) is DSSP v4 only — it arrived in mkdssp 4.0.0 and older versions never emit
# it. The table covers every code v4 can assign; anything unrecognised falls through the `.get`
# default at the call site and becomes C, so a future DSSP code would be silently called coil.
SS8_TO_SS3 = {
    "H": "H", "G": "H", "I": "H",   # α / 3-10 / π helix
    "E": "E", "B": "E",             # β-strand / β-bridge
    "T": "C", "S": "C", "-": "C", "P": "C", " ": "C",  # turn / bend / PPII / coil
}

# --- MaxASA reference tables (Å²), keyed by 1-letter AA -------------------------------
# Tien et al. 2013, PLoS ONE 8(11):e80635, Table 1 (theoretical & empirical); Rost & Sander
# 1994, Proteins 20:216-226. Record which was used — mean |ΔRSA| is 0.012 vs Tien-empirical
# and 0.047 vs Rost & Sander, the latter moving 17% of residues past 0.10 —
# tables. (The `sander_rost1994` key inverts that paper's author order; kept as-is because
# renaming it would break existing configs. Full citations in CITATION.cff.)

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


#: Seconds to wait for `mkdssp --version`. The real probe answers instantly; this exists so a
#: wedged binary fails fast instead of hanging the caller.
_VERSION_PROBE_TIMEOUT = 30


def _detect_version(exe: str) -> str:
    """Parse `mkdssp --version` → semantic version string (default 4.x if unparseable).

    Raises FileNotFoundError, via `_missing_dssp_error`, when the executable cannot be
    launched at all -- a different condition from an unparseable banner, and the one case
    where "assume 4.x and carry on" would produce a misleading downstream failure."""
    try:
        # `timeout` and a closed stdin, because this preflight runs before every uncached
        # `run_dssp`: a binary that hangs or waits on input would otherwise block the caller
        # forever with no error, which is indistinguishable from slow work from the outside.
        out = subprocess.check_output([exe, "--version"], text=True, stderr=subprocess.STDOUT,
                                      timeout=_VERSION_PROBE_TIMEOUT,
                                      stdin=subprocess.DEVNULL)
        m = re.search(r"(\d+\.\d+\.\d+)", out) or re.search(r"(\d+\.\d+)", out)
        if m:
            v = m.group(1)
            return v if v.count(".") == 2 else v + ".0"
    except OSError as exc:
        # Not a parse failure -- the binary could not be launched at all. `OSError` rather than
        # the two obvious subclasses, because there are more than two: ENOENT for an absent
        # binary or a missing interpreter behind a shebang, EACCES for a file that is present
        # but not executable and for a directory, ENOTDIR for a path leading *through* a regular
        # file (a plausible typo in the `dssp.executable` leaf this error tells the reader to
        # edit), and ENOEXEC for a present, executable binary of the wrong architecture -- an
        # x86_64 build on an arm64 machine, which raises plain `OSError`. Every one of them would
        # otherwise fall to the handler below and warn that a version banner could not be parsed,
        # sending the reader after output that was never produced. Chained rather than
        # suppressed, so a caller that wants errno/filename can reach the original via __cause__.
        raise _missing_dssp_error(exe) from exc
    except subprocess.SubprocessError:
        # The binary launched and misbehaved: a non-zero exit (`CalledProcessError`) or a probe
        # that outran its timeout (`TimeoutExpired`). Either way it is an unparseable banner
        # rather than a missing tool, so it belongs with the warning below.
        pass
    # assume modern mkdssp; Biopython then passes --output-format=dssp. Warn, because a
    # real mkdssp 3.x whose banner didn't parse would be driven with a v4-only flag.
    warnings.warn(
        f"Could not parse `{exe} --version`; assuming mkdssp 4.x (--output-format=dssp). "
        "If this is mkdssp 3.x, set dssp.executable explicitly or upgrade.",
        RuntimeWarning, stacklevel=2,
    )
    return "4.0.0"


def _missing_dssp_error(exe: str) -> FileNotFoundError:
    """The actionable error for an absent DSSP binary.

    DSSP is an external program, not a Python package, so it cannot be an extra and pip
    cannot supply it -- which is why this says how to install it rather than naming one.
    Without it Biopython raises a bare FileNotFoundError from inside its own call stack,
    which says neither what needs the binary, nor what still works, nor where to get it --
    and names it wrongly whenever `dssp.executable` is set, since Biopython retries under its
    own hardcoded `mkdssp` when the configured executable is not found.
    """
    return FileNotFoundError(
        f"foldenv needs the {exe!r} binary (DSSP; v4 preferred, 2.x/3.x work) for secondary "
        f"structure and RSA, and it "
        f"could not be launched. It may be absent from PATH, present but not executable, a "
        f"script whose interpreter is missing, or a binary built for another architecture. "
        f"It is an external program, not a Python "
        f"package, so pip cannot install it:\n"
        f"  macOS        brew tap brewsci/bio && brew install brewsci/bio/dssp\n"
        f"  Linux/conda  conda install -c conda-forge -c bioconda dssp\n"
        f"  Debian       apt-get install dssp   (provides mkdssp; 2.x/3.x work too)\n"
        f"Set the D6 config leaf `dssp.executable` if it is installed under another name.\n"
        f"Everything that reports secondary structure or RSA needs it: "
        f"`get_dssp`, `get_structural_context`, `structural_profile`, `tool.invoke`/`tool.call`, "
        f"`analysis.summarize`/`analysis.functional_site_stats` and "
        f"`validation.crystal_crosscheck` all resolve those first. `get_structure`, "
        f"`get_sequence`, `get_contacts`, `fetch_structure` and `clear_cache` do not."
    )


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

    # `_detect_version` is the single guard: it launches the binary before Biopython does and
    # turns an OS-level launch failure into `_missing_dssp_error`. Without it Biopython raises a
    # bare FileNotFoundError three frames below the caller and attributed to Bio/PDB/DSSP.py --
    # which reads as a Biopython bug rather than a missing external tool. Worse, the name it
    # reports is Biopython's hardcoded `mkdssp` rather than the configured one, because it
    # retries under that name when the configured executable is not found. A `which()` preflight
    # here would be redundant with the launch this makes anyway, and could not be exercised
    # separately from it.
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
