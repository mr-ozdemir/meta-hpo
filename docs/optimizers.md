# Optimizers

Every optimizer searches the unit cube `[0, 1]^d` of a `Space` and offers the same interface: `ask()` and
`tell()`, `best_params` and `best_score`. All take `direction`, `seed` and `max_tries`; pass one to
`optimize()` with `optimizer=`:

```python
opt = mh.PSOOptimizer(space, direction="max", swarm_size=20, seed=1)
study = mh.optimize(train, space, direction="max", max_evals=300, optimizer=opt)
```

| Optimizer | Batch per `ask()` | Main settings |
|---|---|---|
| `ABCOptimizer` | the initial population, then one candidate | `colony_size=10`, `limit=15` |
| `PSOOptimizer` | the whole swarm, every iteration | `swarm_size=20`, `inertia=0.7298`, `cognitive=1.49618`, `social=1.49618`, `max_velocity=1.0` |
| `BBPSOOptimizer` | the initial population, then one candidate | `swarm_size=20`, `target`, `crossover=0.5`, `q_alpha=0.1`, `q_gamma=0.9` |

## ABCOptimizer

The Artificial Bee Colony of Karaboga & Basturk (2007), the default optimizer. The first `ask()` returns the
initial population, `colony_size // 2` candidates; every later `ask()` returns one candidate. A food source
that does not improve for `limit` trials is abandoned and replaced by a scout.

meta-hpo follows the phase order of the reference ABC: employed bees, then onlooker bees, then at most one
scout per cycle. It deviates on purpose in four points:

- It searches the unit cube `[0, 1]^d` and decodes each point to typed parameters.
- It accepts a neighbour by comparing raw minimised scores, not fitness values.
- It never hands out a parameter set again while it can still find a new one.
- It treats NaN and infinity as the worst score.

## PSOOptimizer

`PSOOptimizer` moves every particle once per iteration with the velocity update of Shi & Eberhart (1998).
Its defaults are the constriction constants of Clerc & Kennedy (2002); `max_velocity` is measured in widths
of the unit cube. A coordinate that leaves the cube stops on the wall with zero velocity. Because a whole
iteration is one batch, the candidates of an `ask()` can be trained in parallel.

## BBPSOOptimizer

`BBPSOOptimizer` follows Zhang, Lim & Liu (2023): particles move without a velocity, guided by the best
point, a second remote leader and a better neighbour, and cycle through five search operations when they
stop improving. After every iteration the leader and four other particles are refined by a root-finding
step that Q-learning chooses. These steps aim at `target`, the best score your function can return: 1.0
by default for `direction="max"` (accuracy, F1) and 0.0 for `direction="min"` (a loss). Set it when
your score has another bound.

## References

- Karaboga D., Basturk B. (2007). A powerful and efficient algorithm for numerical function optimization:
  artificial bee colony (ABC) algorithm. Journal of Global Optimization 39, 459–471.
  [doi:10.1007/s10898-007-9149-x](https://doi.org/10.1007/s10898-007-9149-x)
- The official ABC page, with reference implementations: <https://abc.erciyes.edu.tr/>
- Kennedy J., Eberhart R. (1995). Particle swarm optimization. Proceedings of IEEE International Conference
  on Neural Networks, 1942–1948. [doi:10.1109/ICNN.1995.488968](https://doi.org/10.1109/ICNN.1995.488968)
- Shi Y., Eberhart R. (1998). A modified particle swarm optimizer. IEEE International Conference on
  Evolutionary Computation, 69–73. [doi:10.1109/ICEC.1998.699146](https://doi.org/10.1109/ICEC.1998.699146)
- Clerc M., Kennedy J. (2002). The particle swarm: explosion, stability, and convergence in a
  multidimensional complex space. IEEE Transactions on Evolutionary Computation 6(1), 58–73.
  [doi:10.1109/4235.985692](https://doi.org/10.1109/4235.985692)
- Kennedy J. (2003). Bare bones particle swarms. IEEE Swarm Intelligence Symposium, 80–87.
- Zhang L., Lim C. P., Liu C. (2023). Enhanced bare-bones particle swarm optimization based evolving deep
  neural networks. Expert Systems with Applications 230, 120642.
  [doi:10.1016/j.eswa.2023.120642](https://doi.org/10.1016/j.eswa.2023.120642)
- Oliveira I. F. D., Takahashi R. H. C. (2020). An enhancement of the bisection method average
  performance preserving minmax optimality. ACM Transactions on Mathematical Software 47(1), 5.
  [doi:10.1145/3423597](https://doi.org/10.1145/3423597)
