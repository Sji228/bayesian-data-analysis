---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

+++ {"editable": true, "slideshow": {"slide_type": ""}}

# Model selection

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import numpy as np
import scipy
import pymc as pm
import arviz as az
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
print(f"Running on PyMC version {pm.__version__} and ArviZ version {az.__version__}")
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(8,6)
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Your aim in this assignment is to find the model that best fits the data

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
data= np.loadtxt('data.txt')
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

The data is from an discrete distribution. You can use the function below to plot the histogram.

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
def plot_discrete_hist(data, ax = None, **kwargs):
    if ax is None:
        ax =plt.gca()
    min, max = data.min(), data.max()
    n_bins=int(max-min+1)
    return ax.hist(data, bins=n_bins, range=(min-0.5, max+0.5), **kwargs)
```

```{code-cell} ipython3
---
editable: true
slideshow:
  slide_type: ''
---
fig, ax = plt.subplots()
plot_discrete_hist(data, ax=ax);
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 1

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Check if the data can be explained by the binomial distribution with $n=35$. Make a posterior predictive check.

```{code-cell} ipython3
tune = 1000
draws = 8000
```

```{code-cell} ipython3
with pm.Model() as binom_35:
    p   = pm.Beta("p", alpha=1, beta=1)
    obs = pm.Binomial("obs", n=35, p=p, observed=data)

    traceb35 = pm.sample(
        tune = 1000,
        draws = 8000,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )
    ppcb35 = pm.sample_posterior_predictive(
        traceb35,
        var_names=["obs"],
        extend_inferencedata=True,   
        random_seed=42,
    )
```

```{code-cell} ipython3
x = np.arange(len(data))                      
pp_samples = ppcb35.posterior_predictive.obs.values.ravel()

y_sim = pp_samples[: len(data)]
plt.scatter(x, data,     label="Dane oryginalne")
plt.scatter(x, y_sim,    label="Dane symulowane (PPC)", alpha=0.7)
plt.legend()
plt.show()
```

```{code-cell} ipython3
pp_samples = ppcb35.posterior_predictive.obs.values.ravel()
plt.hist(data,        bins=100, histtype="step", density=True, label="y (oryginał)")
plt.hist(pp_samples,  bins=100, histtype="step", density=True, label="ỹ (PPC)")

plt.legend()
plt.show()
```

```{code-cell} ipython3
#Nasza data i data syntetyczna wyglądają dojść podobnie, ale sprawdzimy jeszcze możliwości zyskania minimów maksimów i std 
```

```{code-cell} ipython3
#Min
pps35 = traceb35.posterior_predictive["obs"].stack(z=("chain", "draw")).transpose()

plt.hist(np.min(pps35, axis=1), histtype="step", density=True, bins=50)
plt.xlabel("min(obs)")
plt.axvline(data.min(), color="black")      
plt.show()
```

```{code-cell} ipython3
#Max
plt.hist(np.max(pps35, axis=1), histtype='step', density=True, bins=50);
plt.xlabel("max(obs)")
plt.axvline(data.max(), color='black');
```

```{code-cell} ipython3
plt.hist(np.std(pps35, axis=1), histtype='step', density=True, bins=50);
plt.xlabel("std(obs)")
plt.axvline(data.std(), color='black');
```

```{code-cell} ipython3
#Z danych syntetycznych można zyskać maksimum, minimum ale nie można zyskać std
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 2

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for binomial distribution with $n=45$.

```{code-cell} ipython3
with pm.Model() as binom_45:
    # prior
    p = pm.Beta("p", alpha=1, beta=1)

    # likelihood
    obs = pm.Binomial("obs", n=45, p=p, observed=data)

    # posterior
    traceb45 = pm.sample(
    return_inferencedata=True,
    idata_kwargs={"log_likelihood": True},  
    random_seed=42,
    target_accept=0.9
    )


    # posterior predictive
    ppcb45 = pm.sample_posterior_predictive(
        traceb45,
        var_names=["obs"],
        extend_inferencedata=True,
        random_seed=42,
    )
```

```{code-cell} ipython3
x = np.arange(len(data))                      
pp_samples = ppcb45.posterior_predictive.obs.values.ravel()

y_sim = pp_samples[: len(data)]
plt.scatter(x, data,     label="Dane oryginalne")
plt.scatter(x, y_sim,    label="Dane symulowane (PPC)", alpha=0.7)
plt.legend()
plt.show()
```

```{code-cell} ipython3
pp_samples45 = ppcb45.posterior_predictive.obs.values.ravel()
plt.hist(data,        bins=100, histtype="step", density=True, label="y (oryginał)")
plt.hist(pp_samples45,  bins=100, histtype="step", density=True, label="ỹ (PPC)")

plt.legend()
plt.show()
```

```{code-cell} ipython3
pps45 = traceb45.posterior_predictive["obs"].stack(z=("chain", "draw")).transpose()

plt.hist(np.min(pps45, axis=1), histtype="step", density=True, bins=50)
plt.xlabel("min(obs)")
plt.axvline(data.min(), color="black")      
plt.show()
```

```{code-cell} ipython3
plt.hist(np.max(pps45, axis=1), histtype='step', density=True, bins=50);
plt.xlabel("max(obs)")
plt.axvline(data.max(), color='black');
```

```{code-cell} ipython3
plt.hist(np.std(pps45, axis=1), histtype='step', density=True, bins=50);
plt.xlabel("std(obs)")
plt.axvline(data.std(), color='black');
```

```{code-cell} ipython3
#Z danych syntetycznych można zyskać maksimum, minimum ale nie można zyskać std
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 3

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Repeat the same for poisson distribution.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

$$p(k)=e^{-\lambda}\frac{\lambda^k}{k!}$$

+++ {"editable": true, "slideshow": {"slide_type": ""}}

 Use $\Gamma$ distribution for prior on the $\lambda$ parameter of the distribution.

```{code-cell} ipython3
with pm.Model() as model_pois:
    lam = pm.Gamma("lam", alpha=1.0, beta=1.0)   
    obs = pm.Poisson("obs", mu=lam, observed=data)

    trace_pois = pm.sample(
        tune=tune,
        draws=draws,
        chains=4,
        target_accept=0.9,
        random_seed=42,
        return_inferencedata=True,
        idata_kwargs={"log_likelihood": True},
    )

    ppc_pois = pm.sample_posterior_predictive(
        trace_pois,
        var_names=["obs"],
        extend_inferencedata=True,
        random_seed=42,
    )
```

```{code-cell} ipython3
x = np.arange(len(data))
pp_samples = ppc_pois.posterior_predictive.obs.values.ravel()
y_sim = pp_samples[: len(data)]          

plt.figure(figsize=(6, 4))
plt.scatter(x, data,  label="Dane oryginalne")
plt.scatter(x, y_sim, label="Dane symulowane (PPC)", alpha=0.7)
plt.legend()
plt.show()
```

```{code-cell} ipython3
pps = ppc_pois.posterior_predictive["obs"].stack(z=("chain", "draw")).transpose()
plt.hist(data,        bins=100, histtype="step", density=True, label="y (oryginał)")
plt.hist(pps,  bins=100, histtype="step", density=True, label="ỹ (PPC)")

plt.legend()
plt.show()
```

```{code-cell} ipython3
plt.hist(np.min(pps, axis=1), histtype="step", density=True, bins=50)
plt.axvline(data.min(), color="black")
plt.xlabel("min(obs)")
plt.title("Posterior predictive: minimum")
plt.show()
```

```{code-cell} ipython3
plt.hist(np.max(pps, axis=1), histtype="step", density=True, bins=50)
plt.axvline(data.max(), color="black")
plt.xlabel("max(obs)")
plt.title("Posterior predictive: maksimum")
plt.show()
```

```{code-cell} ipython3
plt.hist(np.std(pps, axis=1), histtype="step", density=True, bins=50)
plt.axvline(data.std(), color="black")
plt.xlabel("std(obs)")
plt.title("Posterior predictive: odchylenie std.")
plt.show()
```

```{code-cell} ipython3
#Z danych syntetycznych można zyskać maksimum, minimum i już można zyskać std, więc ten model jest najlepszy.
```

+++ {"editable": true, "slideshow": {"slide_type": ""}}

### Problem 4

+++ {"editable": true, "slideshow": {"slide_type": ""}}

Compare the three models using the "leave one out" cross validation.

+++ {"editable": true, "slideshow": {"slide_type": ""}}

If you can, explain the results :)

```{code-cell} ipython3
comp_df = az.compare({
    "binom_35": traceb35,
    "binom_45": traceb45,
    "poisson": ppc_pois
})
az.plot_compare(comp_df);
```

```{code-cell} ipython3

```

```{code-cell} ipython3

```
