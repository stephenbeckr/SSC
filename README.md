
# Sparse Subspace Clustering (SSC) algorithms

This repository contains Matlab code to implement ADMM and proximal gradient algorithms to solve the SSC clustering problem (several variants, including variants for affine subspaces). The SSC model we work with is based on the well-known work of [Elhamifar and Vidal's Sparse "Subspace Clustering: Algorithm, Theory, and Applications" (IEEE Trans. on PAMI)](https://scholar.google.com/scholar?cluster=7262850065108933522&hl=en&as_sdt=0,6&as_vis=1)

## Why use this code?

The new ADMM code is much faster, as it scales like *O(n^2)* instead of *O(n^3)*

![Scaling](figs/Fig4.jpg?raw=true "Good Scaling")

The proximal gradient code doesn't have the extra `rho` parameter that ADMM algorithms need. The ADMM algorithms are quite sensitive to this parameter, as the following experiment shows:

![Parameters](figs/Fig2.jpg?raw=true "Parameter Issues")

## Authors
The authors are [Stephen Becker](http://amath.colorado.edu/faculty/becker/) and [Farhad Pourkamali-Anaraki](http://www.pourkamali.com/) (University of Colorado Applied Math)

This README from APril 2018. Thanks to https://stackedit.io/app for editing markup
