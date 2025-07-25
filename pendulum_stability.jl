using SumOfSquares
using DynamicPolynomials
using GLMakie
using LinearAlgebra

using MosekTools, Mosek

# Pendulum's dynamics (including equilibrium point)
g = 9.81
m = 1
l = 1
b = 0.1

@polyvar s c ω

x = [s, c, ω]
f = [c * ω, -s * ω, (-g * m * l * s - b * ω) / (m * l^2)]

x0 = [0, 1, 0]

# SOS Model
model = SOSModel(Mosek.Optimizer)

## V(x)
m = monomials(x, 0:1)
@variable(model, V, SOSPoly(m))

## domain of interest
S = @set s^2 + c^2 == 1

## term for enforcing strict positivity
pos = 1e-4 * dot(x - x0, x - x0)

## Constraints
### strictly positive V
@constraint(model, V >= pos)

### negative V̇
@constraint(model, -dot(differentiate(V, x), f) >= pos * s^2)

### V(x0) = 0
@constraint(model, V(s => x0[1], c => x0[2], ω => x0[3]) == 0)

## Optimization
optimize!(model)

# Plotting
θs = -2*pi:1e-2*pi:2*pi
ωs = -3:1e-2:3

Θ = [x_ for x_ in θs, _ in ωs]
Ω = [y_ for _ in θs, y_ in ωs]

opt_V = value(V)
Z = map((θ_, ω_) -> opt_V(s => sin(θ_), c => cos(θ_), ω => ω_), Θ, Ω)

contour(Θ, Ω, Z, levels=10, labels=true)

