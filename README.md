# TENSOR-KSS

Companion code for the paper

> **Non-asymptotic Tail Bounds for the Kostlan–Shub–Smale Field: Tensor PCA and Spherical _k_-Spin Complexity**
> Jean-Marc Azaïs, Federico Dalmao, Yohann De Castro.

The paper bounds the tail of the supremum of the Kostlan–Shub–Smale random field on the sphere through a Kac–Rice reduction to a shifted Gaussian Orthogonal Ensemble, and uses it to obtain (i) a non-asymptotic error bound for the profile maximum-likelihood estimator in spiked Tensor PCA, and (ii) a non-asymptotic two-sided bracketing of the annealed complexity of the spherical _k_-spin Hamiltonian. This repository contains the two notebooks that **certify every constant and closed form in the paper** and that **reproduce its figures**.

## Contents

| File | Description |
|---|---|
| `Verification_Tensor_KSS_KacRice.ipynb` | Symbolic + numerical certification of the manuscript. Re-derives every load-bearing identity, constant, closed form, and inequality. **450/450 checks pass** across 14 sections — **430 symbolic** (SymPy, proved exactly, no numeric fallback) and **20 adaptive quadrature** (the closed-form `δ_exact` against the Kac–Rice integral, to ~10⁻⁸–10⁻¹⁶ relative error). |
| `figures_article.ipynb` | Reproduces the figures: the tail-bound comparison `δ_exact/δ_IMF/δ_SMF/δ_SM` vs. the baseline (`delta_min_comparison.pdf`), the two-sided LRT threshold inversion (`threshold_inversion_panels.pdf`), and a supplementary normalized-ratio plot (`delta_ratio_baseline.pdf`). |
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
```

A clean run of the verification notebook ends with

```
OVERALL: 450/450 checks passed across 14 sections.
  symbolic (SymPy)    : 430 passed
  adaptive quadrature : 20 passed
```

## What is verified

The verification notebook covers, among others: the Mehta–Fyodorov representation and the GOE one-point (Fyodorov/Mehta) identities; the four-piece closed form of `δ_exact` and the Hermite-tail recurrences `J_mᵝ`, `K_jᵝ`; the IMF / SMF / SM bounds and their nesting; the prefactor asymptotics; the Ben Arous–Dembo–Guionnet spectral-radius lemma and the Step-4 amplifier constants (`C₁=64`, `C₂=8`, `C_amp=8√2`); the spherical _k_-spin threshold constants (`ρE∞=√2`, `E_BDG`) and the Auffinger–Ben Arous–Černý complexity match `Θₖ`.

## Citation

```bibtex
@article{azais2026nonasymptotic,
  title  = {Non-asymptotic Tail Bounds for the Kostlan--Shub--Smale Field:
            Tensor PCA and Spherical $k$-Spin Complexity},
  author = {Aza{\"\i}s, Jean-Marc and Dalmao, Federico and De Castro, Yohann},
  year   = {2026}
}
```

A companion paper on the exact (_t_-)spacing test for the same model is
[Azaïs, Dalmao, De Castro, *Second Maximum of a Gaussian Random Field and Exact (t-)Spacing test*, arXiv:2406.18397](https://arxiv.org/abs/2406.18397).

## License

Released under the [MIT License](LICENSE).
