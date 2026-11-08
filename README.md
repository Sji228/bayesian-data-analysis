# Bayesian Data Analysis

My solutions to the assignments from the Bayesian Data Analysis course at Jagiellonian University (spring 2025), done together with Witold ([@Witek3105](https://github.com/Witek3105)). They cover parameter estimation on a grid, MCMC with PyMC, posterior predictive checks and model comparison.

## Projects

| | Problem | Methods |
|---|---|---|
| [**Lighthouse, 1D**](01-lighthouse/1_lighthouse_1d.ipynb) | A lighthouse flashes at random angles; only the positions where flashes hit the shore are recorded. Find where the lighthouse is. | Cauchy likelihood, grid posterior, MAP, sequential updating as data arrives |
| [**Lighthouse, 2D**](01-lighthouse/2_lighthouse_2d.ipynb) | Same, but the distance from the shore is unknown too. | 2D grid posterior, marginal distributions, log-space computation |
| [**Lighthouse with MCMC**](01-lighthouse/3_lighthouse_mcmc.ipynb) | Both lighthouse problems again, solved by sampling. | PyMC, MAP, NUTS sampling, posterior summaries with 95% HDI |
| [**Model selection**](02-model-selection/model_selection.ipynb) | Which distribution generated a set of counts: binomial (n=35), binomial (n=45) or Poisson? | Posterior predictive checks on min, max and std; leave-one-out cross-validation |
| [**Quadratic regression**](03-regression/quadratic_regression.ipynb) | Fit a second-degree polynomial with unknown noise. | Least squares (`curve_fit`) vs. Bayesian fit in PyMC, parameter uncertainties |

The `.ipynb` files contain the outputs and plots, so they can be read directly on GitHub. Each one also has a [MyST Markdown](https://jupytext.readthedocs.io/en/latest/formats-markdown.html) twin (`.md`), the format the course used for version control.

## Running it

```bash
pip install -r requirements.txt
jupyter lab
```

## Credits

Problem statements, datasets and `lighthouse.py` come from the course materials by Piotr Białas. The solutions are ours.
