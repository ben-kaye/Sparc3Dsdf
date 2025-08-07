# Implementing Sparc3d

https://lizhihao6.github.io/Sparc3D/

Sparc3D have not as of 2025-August-07 released their code, so I have been working on my own implementation. Please join my in recreating their methodology for the good of the Computer vision community! I will be taking pull requests.

For now I am implementing just the SDF computations described.

## Major TODOs
- [ ] Port marching cubes from https://github.com/nv-tlabs/FlexiCubes/blob/main/flexicubes.py to use the Sparcubes representation and export a mesh
- [ ] Differentiable rendering to refine Sparcubes
- [ ] Fix `sparc3d_sdf.sparc3d.get_displacements` to calculate only within the "active mask"