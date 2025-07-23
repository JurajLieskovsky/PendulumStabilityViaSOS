using SumOfSquares
using DynamicPolynomials
using GLMakie
using LinearAlgebra

using MosekTools, Mosek

# Pendulum's dynamics (including equilibrium point)
k = 1   # g * m * l / m * l^2
b = 0.1 # b / m * l^2

@polyvar s c ω

x = [s, c, ω]
f = [c * ω, -s * ω, -k * s - b * ω]

x0 = [0, 1, 0]

# SOS Model
model = SOSModel(Mosek.Optimizer)

## V(x)
m = monomials(x, 0:2)
@variable(model, V, PolyJuMP.Poly(m))

## domain of interest
S = @set s^2 + c^2 == 1

## term for enforcing strict positivity
pos = 1e-4 * dot(x - x0, x - x0)

## Constraints
### strictly positive V
@constraint(model, V >= pos, domain = S)

### negative V̇
@constraint(model, -dot(differentiate(V, x), f) >= pos * s^2, domain = S)

### V(x0) = 0
@constraint(model, V(s => x0[1], c => x0[2], ω => x0[3]) == 0)

## Optimization
optimize!(model)

# Plotting
θs = -2*pi:1e-2*pi:2*pi
ωs = -2.5:1e-2:2.5

Θ = [x_ for x_ in θs, _ in ωs]
Ω = [y_ for _ in θs, y_ in ωs]

Z = map((θ_, ω_) -> value(V)(s => sin(θ_), c => cos(θ_), ω => ω_), Θ, Ω)

contour(Θ, Ω, Z, levels=10, labels=true)

