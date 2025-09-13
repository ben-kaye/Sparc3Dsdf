# Hermes (formerly Sparc3D SDF)

A fast SDF preprocessing library for meshes built on the ops provided in Kaolin, and referencing the approach described in Sparc3D.

This originally a reference implementation using Kaolin based on the Sign distance field preprocessing described in the [Sparc3D paper / arXiv](https://arxiv.org/abs/2505.14521)  (which had/has not released their code, 2025-09-13).

# Installation:
See `INSTALL.md`

# Usage
Example script: `scripts/sdf.py`. 

```
python scripts/sdf.py -i assets/plane.obj --N 1024 -o plane_1024.obj
```

Calculates the SDF for the provided `assets/plane.obj` from Objaverse at $1024^3$ in under $30$ s with less than 16 GB VRAM.


## Open TODOs
[ ] Render refinement?
[ ] Improve the flood-fill extraction, see https://github.com/isl-org/unifi3d/blob/374201a23f2a3c36c8e595eec9431b77df8c08fa/scripts/compare_sdf.py#L31

