"""Crystal cross-check: AF-predicted vs experimental RSA + secondary structure.

Quantifies how much AlphaFold's per-residue RSA/SS agree with an experimental structure of
the same protein — a sanity bound on the AF-derived structural fields (esp. AF's distortion
of low-confidence loops). The AF↔experimental residue mapping is built by **sequence
alignment**, not by trusting residue numbers, because experimental "author" numbering (e.g.
TEM-1's Ambler numbering in 1BTL) differs from UniProt/AF numbering.

`crystal_crosscheck("P62593", "1BTL")` → CrosscheckReport with SS agreement + RSA
Pearson/MAE over the residues present in both.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass

from .constants import three2one

from . import config as _config
from . import context as _ctx
from .dssp import run_dssp
from .fetch import fetch_experimental_structure


@dataclass
class CrosscheckReport:
    uniprot_id: str
    pdb_id: str
    chain_id: str
    n_compared: int          # residues present (with DSSP) in both structures
    ss3_agreement: float     # fraction of compared residues with matching H/E/C
    rsa_pearson: float       # Pearson r of RSA (AF vs experimental)
    rsa_mae: float           # mean |ΔRSA|


def _chain_seq_and_resnums(structure, chain_id: str | None):
    """Ordered (sequence, parallel author-resnums, chain_id) for one chain, standard AAs only."""
    chain = (
        structure[0][chain_id] if chain_id is not None
        else next(structure[0].get_chains())
    )
    seq, nums, has_icode = [], [], False
    for r in chain:
        if r.id[0].strip():  # skip HETATM/water
            continue
        if r.id[2].strip():  # insertion code present
            has_icode = True
        aa = three2one.get(r.resname.strip().upper())
        if aa is None:  # skip non-standard for a clean alignment
            continue
        seq.append(aa)
        nums.append(r.id[1])
    if has_icode:
        # We key residues by number only (fine for AF and most crystals); insertion codes
        # (proteases/antibodies) can share a number and would mismap. Warn rather than fail.
        warnings.warn(
            f"chain {chain.id} has insertion codes — the cross-check keys residues by number "
            "only and may mismap them for this structure.",
            stacklevel=2,
        )
    return "".join(seq), nums, chain.id


def _map_exp_to_af(af_seq: str, exp_seq: str, exp_nums: list[int]) -> dict[int, int]:
    """Global-align exp→AF; return {experimental_resnum: af_position (1-based)}."""
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = -10.0
    aligner.extend_gap_score = -0.5
    aln = aligner.align(af_seq, exp_seq)[0]  # target=AF, query=exp

    mapping: dict[int, int] = {}
    af_blocks, exp_blocks = aln.aligned  # aligned, gap-free segments
    for (t0, t1), (q0, q1) in zip(af_blocks, exp_blocks):
        for k in range(t1 - t0):
            mapping[int(exp_nums[q0 + k])] = int(t0 + k) + 1  # AF pos is 1-based
    return mapping


def crystal_crosscheck(
    uniprot_id: str,
    pdb_id: str,
    chain_id: str | None = None,
    cfg: dict | None = None,
) -> CrosscheckReport:
    """Compare AF vs experimental RSA/SS for `uniprot_id` against PDB `pdb_id`."""
    cfg = cfg or _config.load()

    af_dssp = _ctx.get_dssp(uniprot_id, cfg)          # {af_resnum: ResidueDSSP}
    af_seq = _ctx.get_sequence(uniprot_id, cfg)

    cif_path, exp_struct = fetch_experimental_structure(pdb_id, cfg["cache"]["dir"])
    exp_seq, exp_nums, chain_used = _chain_seq_and_resnums(exp_struct, chain_id)
    exp_dssp = run_dssp(
        cif_path, chain_id=chain_used,
        exe=cfg["dssp"]["executable"], max_asa_table=cfg["rsa"]["max_asa_table"],
    )

    exp_to_af = _map_exp_to_af(af_seq, exp_seq, exp_nums)

    ss_match, n_compared, af_rsa, exp_rsa = 0, 0, [], []
    for exp_num, af_pos in exp_to_af.items():
        a, e = af_dssp.get(af_pos), exp_dssp.get(exp_num)
        if a is None or e is None:
            continue
        n_compared += 1
        ss_match += a.ss3 == e.ss3
        # compare RSA only where both are defined (non-standard → NaN)
        if a.rsa == a.rsa and e.rsa == e.rsa:  # not NaN
            af_rsa.append(a.rsa)
            exp_rsa.append(e.rsa)

    if n_compared == 0:
        raise ValueError(f"no comparable residues for {uniprot_id} vs {pdb_id}")

    import numpy as np

    af_arr, exp_arr = np.array(af_rsa), np.array(exp_rsa)
    pearson = float(np.corrcoef(af_arr, exp_arr)[0, 1]) if len(af_rsa) >= 2 else float("nan")
    mae = float(np.mean(np.abs(af_arr - exp_arr))) if af_rsa else float("nan")

    return CrosscheckReport(
        uniprot_id=uniprot_id,
        pdb_id=pdb_id.upper(),
        chain_id=chain_used,
        n_compared=n_compared,
        ss3_agreement=ss_match / n_compared,
        rsa_pearson=pearson,
        rsa_mae=mae,
    )
