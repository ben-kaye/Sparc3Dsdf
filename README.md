# Implementing Sparc3d

https://lizhihao6.github.io/Sparc3D/

Sparc3D have not as of 2025-August-07 released their code, so I have been working on my own implementation. Please join my in recreating their methodology for the good of the Computer vision community! I will be taking pull requests.

I have limited scope to just the SDF pre-processing as I wanted to learn how to complete these calculations!

# Installation:
See `INSTALL.md`

# Usage
Example script: `scripts/sdf_mesh.py`.  Calculates the SDF for the provided `assets/plane.obj` from Objaverse at $1024^3$ in under $30$ s with less than 16 GB VRAM.

## Also included..
To visualise the flood fill algorithm I also implemented a voxel version of marching cubes. Example is under `scripts/voxel_mesh.py`


## TODOs
[X] Port marching cubes from https://github.com/nv-tlabs/FlexiCubes/blob/main/flexicubes.py to be sparse aware\
[ ] Include render refinement\
[ ] Triangle correction described in paper
