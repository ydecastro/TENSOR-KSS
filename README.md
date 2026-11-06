# TENSOR-KSS

Companion code for two papers on the Kostlan–Shub–Smale field and the spiked tensor model, by Jean-Marc Azaïs, Federico Dalmao and Yohann De Castro:

> **Non-asymptotic First- and Second-Order Tail Brackets for the Kostlan–Shub–Smale Field and Spherical _k_-Spin Complexity** (the KSS paper). A two-sided bracket $(1-\varepsilon(u))\,\delta_*(u)\le\mathbb P\{\sup X>u\}\le\delta_*(u)$ for the supremum of the Kostlan–Shub–Smale field on the sphere at every finite $(k,d)$, the Kac–Rice integral $\delta_*$ in closed form as a finite Hermite sum with two explicit relaxations, and a non-asymptotic bracket for the annealed complexity of the spherical _k_-spin Hamiltonian.

> **The _t_-spacing test for a spiked tensor** (the statistics paper). Power of the exact, variance-free _t_-spacing test, asymptotic minimaxity of the argmax with the sharp constant $(d-1)/(k\lambda^2)$, and the deficiency bracket at high signal.

Both papers descend from the manuscript *Non-asymptotic Tail Bounds for the Kostlan–Shub–Smale Field: Tensor PCA and Spherical k-Spin Complexity* (June 2026), which this repository originally accompanied.

## Contents

| File | Description |
|---|---|
| `Verification_Tensor_KSS_KacRice.ipynb` | Symbolic + numerical certification of the June 2026 manuscript: every load-bearing identity, constant, closed form and inequality re-derived (**450/450 checks**, 430 symbolic with SymPy, 20 by adaptive quadrature). The KSS paper keeps the Mehta–Fyodorov representation, the closed form of $\delta_{\mathrm{exact}}$, the IMF and SMF bounds and their nesting and the complexity constants, all certified here; the spectral-method bound $\delta_{\mathrm{SM}}$, the closed-form inversion of the SMF bound and the Ben Arous–Dembo–Guionnet lemma, also certified here, are no longer in the papers (the KSS paper's constant $\beta_d$ of Lemma 2.9 is the one of the June manuscript without its second term $B_d'$, which the manuscript kept only for definiteness). |
| `figures_article.ipynb` | Reproduces the figures: Figure 1 of the KSS paper (`delta_min_comparison.pdf`, the bounds $\delta_{\mathrm{exact}},\delta_{\mathrm{IMF}},\delta_{\mathrm{SMF}}$ against the baseline), a supplementary prefactor-ratio plot (`delta_ratio_baseline.pdf`), and a supplementary threshold-inversion plot (`threshold_inversion_panels.pdf`, inversion of $\delta_{\mathrm{IMF}}$ and $\delta_{\mathrm{exact}}$ at levels $10^{-3}$ and $0.05$; it was Figure 1 of an earlier version of the statistics paper). |
| `table2_rice.py` | Table 2 of the KSS paper: the Rice integral of Lemma 4.4 with Hadamard's bound, evaluated numerically, the explicit bound $\bar P(u)$ of Proposition 4.8, and the two values of $\varepsilon$ of the proof of Theorem 1.1 at the six points $(k,d,u)$ of the table, with the levels where the explicit and the numerical bounds reach 1. Output `table2_rice_output.txt`. |
| `figure2/` | Figure 2 of the KSS paper (`layer_weights_k5.pdf`, the layer weights $w_j$ and the variance $V$ of the helix process for $k=5$): `helix_weights.py` (standard library only) prints the critical points of $w_j$ and writes the data files, `layer_weights_k5.tex` (pgfplots) draws them. |
| `paper2/` | Code for every number of the statistics paper: see `paper2/README.md`. |
| `LICENSE` | MIT License. |

## Requirements

- Python ≥ 3.10
- `numpy`, `scipy`, `sympy`, `matplotlib`, `jupyter`

```bash
pip install numpy scipy sympy matplotlib jupyter
```

## Usage

Open either notebook in Jupyter and run all cells, or execute headlessly:

```bash
jupyter nbconvert --to notebook --execute Verification_Tensor_KSS_KacRice.ipynb
jupyter nbconvert --to notebook --execute figures_article.ipynb
python table2_rice.py
cd figure2 && python helix_weights.py && pdflatex layer_weights_k5.tex
```

A clean run of the verification notebook ends with

```
OVERALL: 450/450 checks passed across 14 sections.
  symbolic (SymPy)    : 430 passed
  adaptive quadrature : 20 passed
```

## Reproducibility

The output files kept in the repository (`delta_ratio_baseline.pdf`,
`threshold_inversion_panels.pdf`, `table2_rice_output.txt`) were produced on 2026-10-23 by the
notebook and the script above with Python 3.14.3, numpy 2.4.2, scipy 1.17.1, sympy 1.14.0, matplotlib 3.10.8.
`delta_min_comparison.pdf` was produced again on 2026-11-06, after the threshold labels of the
figure were moved inside the axes, with Python 3.12.3, numpy 1.26.4, scipy 1.15.2, matplotlib 3.10.0; the curves and the printed numbers
are unchanged. `figure2/layer_weights_k5.pdf` was compiled on 2026-11-06 with pdflatex (TeX Live 2025) from
the data written by `figure2/helix_weights.py`. `delta_min_comparison.pdf` is the file included
as Figure 1 of the KSS paper (identical rendering); the numbers printed by the notebook
(ratios at $u=12$) and by `table2_rice.py` are those of the caption of Figure 1 and of Table 2.

## Conventions

The GOE of size $n$ is in the Mehta normalisation, $\mathrm{Var}(G_{ii})=1$, $\mathrm{Var}(G_{ij})=1/2$, obtained as $(A+A^\top)/2$ from i.i.d. standard Gaussian entries; $\rho^2=k/(2(k-1))$; the Kac–Rice prefactor is $C_{k,d}=2\sqrt\pi\,(k-1)^{(d-1)/2}/\Gamma(d/2)$. The supplementary threshold figure inverts the one-sided bounds: for $k$ odd $\sup|X|=\sup X$, and for $k$ even the KSS paper's Theorem 1.1 halves the critical-point count, so no factor 2 is needed for the two-sided supremum.

## Citation

```bibtex
@unpublished{azais2026kss,
  title  = {Non-asymptotic First- and Second-Order Tail Brackets for the Kostlan--Shub--Smale Field
            and Spherical $k$-Spin Complexity},
  author = {Aza{\"\i}s, Jean-Marc and Dalmao, Federico and De Castro, Yohann},
  year   = {2026}
}

@unpublished{azais2026spacing,
  title  = {The $t$-spacing test for a spiked tensor},
  author = {Aza{\"\i}s, Jean-Marc and Dalmao, Federico and De Castro, Yohann},
  year   = {2026}
}
```

The exact (_t_-)spacing tests themselves are from
[Azaïs, Dalmao, De Castro, *Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test*, arXiv:2406.18397](https://arxiv.org/abs/2406.18397).

## License

Released under the [MIT License](LICENSE).
