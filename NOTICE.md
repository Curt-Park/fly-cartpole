# Notices

## MaleCNS v1.0 connectome (data)

`data/mb_right.npz`, `data/mb_left.npz`, their `_manifest.json` files,
`web/data/cells.json` and `web/data/model.json` are derived from the MaleCNS
v1.0 connectome of the adult male *Drosophila* central nervous system (HHMI
Janelia, Google Research, University of Cambridge, MRC Laboratory of Molecular
Biology and collaborators; Berg et al., Cell 2026), licensed under CC BY 4.0:
https://creativecommons.org/licenses/by/4.0/.

Changes: the left and right mushroom bodies (projection neurons, Kenyon cells,
output neurons) and their PAM/PPL1 dopaminergic inputs were selected and reduced
to synapse-count matrices and cell-body positions; `web/data/model.json` also
holds synapse gains learned in simulation.

Source files:
https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/

## fly-blackjack (method)

The depression rule, the valence derivation from dominant dopaminergic input and
the T-maze style choice of the first design follow fly-blackjack by William
Jones (MIT License):
https://github.com/WilliamJones/fly-blackjack
