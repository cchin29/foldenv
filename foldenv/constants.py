"""Amino-acid tables and the PLM-encoder registry.

Vendored so ``foldenv`` has no import dependency on any external package for these
facts. The amino-acid lookups are standard reference data; ``PLM_ENCODERS`` maps each
supported embedding backend to its HuggingFace Hub id.
"""

# Single letter, three letter, and full amino acid names.
aa_names = (
    ('A', 'ALA', 'alanine'),
    ('R', 'ARG', 'arginine'),
    ('N', 'ASN', 'asparagine'),
    ('D', 'ASP', 'aspartic acid'),
    ('C', 'CYS', 'cysteine'),
    ('E', 'GLU', 'glutamic acid'),
    ('Q', 'GLN', 'glutamine'),
    ('G', 'GLY', 'glycine'),
    ('H', 'HIS', 'histidine'),
    ('I', 'ILE', 'isoleucine'),
    ('L', 'LEU', 'leucine'),
    ('K', 'LYS', 'lysine'),
    ('M', 'MET', 'methionine'),
    ('F', 'PHE', 'phenylalanine'),
    ('P', 'PRO', 'proline'),
    ('S', 'SER', 'serine'),
    ('T', 'THR', 'threonine'),
    ('W', 'TRP', 'tryptophan'),
    ('Y', 'TYR', 'tyrosine'),
    ('V', 'VAL', 'valine'),
    # Extended AAs
    ('B', 'ASX', 'asparagine or aspartic acid'),
    ('Z', 'GLX', 'glutamine or glutamic acid'),
    ('X', 'XAA', 'Any'),
    ('J', 'XLE', 'Leucine or isoleucine'),
)

# Indices of standard amino acids in `aa_names`.
standard_indices = tuple(range(20))

# Single letter codes of standard amino acids.
standard_aas = tuple(aa_names[i][0] for i in standard_indices)
AAs = tuple(sorted(standard_aas))

# aa_to_idx and idx_to_aa
aa2idx = dict(zip(AAs, standard_indices))
idx2aa = {v: k for k, v in aa2idx.items()}

# dictionaries for aa name conversion
one2three = dict(aa_names[i][:2] for i in standard_indices)
three2one = {v: k for k, v in one2three.items()}


# PLM encoders → HuggingFace Hub ids
PLM_ENCODERS = {
    "esm": "facebook/esm2_t36_3B_UR50D",
    "ankh": "ElnaggarLab/ankh-large",
    "esm_35M": "facebook/esm2_t12_35M_UR50D",
    "esm_650M": "facebook/esm2_t33_650M_UR50D",
    "ankh_base": "ElnaggarLab/ankh-base",
    "protbert": "Rostlab/prot_bert",
    "prott5_xl_half": "Rostlab/prot_t5_xl_half_uniref50-enc",
    "prostt5": "Rostlab/ProstT5",
    "esmc_6b": "EvolutionaryScale/esmc-6b-2024-12",  # ESM Cambrian 6B, 2560-dim (needs transformers>=4.57)
    "ankh3_large": "ElnaggarLab/ankh3-large",        # Ankh3-large, T5 encoder, 1536-dim
    "ankh3_xl": "ElnaggarLab/ankh3-xl",              # Ankh3-XL, T5 encoder, 2560-dim
    "saprot": "westlake-repl/SaProt_650M_AF2",       # SaProt 650M, ESM2-650M arch + SA (AA+3Di) vocab, 1280-dim
    "saprot_1.3b": "westlake-repl/SaProt_1.3B_AFDB_OMG_NCBI",  # SaProt 1.3B, ESM arch, 1280-dim
}
