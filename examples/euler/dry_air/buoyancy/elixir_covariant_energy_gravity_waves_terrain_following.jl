###############################################################################
# DGSEM for the covariant Euler equations in total energy formulation
###############################################################################

using OrdinaryDiffEqLowStorageRK, Trixi, TrixiAtmo

###############################################################################
# Spatial discretization

"""
    initial_condition_gravity_waves(x, t,
                                        equations::CovariantEulerEnergyEquations2D)

Test cases for linearized analytical solution by
-  Baldauf, Michael and Brdar, Slavko (2013)
   An analytic solution for linear gravity waves in a channel as a test
   for numerical models using the non-hydrostatic, compressible {E}uler equations
   [DOI: 10.1002/qj.2105] (https://doi.org/10.1002/qj.2105)
"""
function initial_condition_gravity_waves(x, t,
                                         equations::CovariantEulerEnergyEquations2D)
    g = equations.gravity
    c_p = equations.c_p
    c_v = equations.c_v
    # center of perturbation
    x_c = 100_000.0
    a = 5_000
    H = 10_000
    R = c_p - c_v    # gas constant (dry air)
    T0 = 250
    delta = g / (R * T0)
    DeltaT = 0.00
    Tb = DeltaT * sinpi(x[2] / H) * exp(-(x[1] - x_c)^2 / a^2)
    ps = 100_000  # reference pressure
    rhos = ps / (T0 * R)
    rho_b = rhos * (-Tb / T0)
    p = ps * exp(-delta * x[2])
    rho = rhos * exp(-delta * x[2]) + rho_b * exp(-0.5 * delta * x[2])
    v1 = 0
    v2 = 0

    return SVector(rho, v1, v2, p)
end

@inline function geopotential(x, equations::CovariantEulerEnergyEquations2D)
    return equations.gravity * x[2]
end

initial_condition = initial_condition_gravity_waves

equations = CovariantEulerEnergyEquations2D(c_p = 1004,
                                            c_v = 717,
                                            gravity = EARTH_GRAVITATIONAL_ACCELERATION)

###############################################################################
# Build DG solver.

polydeg = 4

dg = DGMulti(element_type = Quad(),
             approximation_type = SBP(),
             surface_flux = flux_lax_friedrichs,
             polydeg = polydeg)

###############################################################################
# Build mesh.


function schaer_mountain(x, H_0, a_e, a_c, x_0)

    return H_0 * exp(-((x-x_0)^2/a_e^2)) * cos(pi * ((x - x_0)/a_c))^2

end

function stretching_function(y, beta)

    return beta * y^2 + (1 - beta) * y

end

L_x = 300_000.0
H_top = 10_000.0
beta = 0.0

h = x -> schaer_mountain(x, 10.0, 10_000.0, 8_000.0, L_x /2)
f_s = y -> stretching_function(y, beta)
f_s_inv = phi -> TrixiAtmo.stretching_function_inverse(phi, beta)

manifold = TerrainFollowingManifold(h, H_top, f_s, f_s_inv)
metric_terms = MetricTermsCovariant(manifold = manifold,
                                    christoffel_symbols = ChristoffelSymbolsAutodiff())


mesh = DGMultiMeshQuadTerrainFollowing2D(dg, 120, 16, L_x, metric_terms)

# Transform the initial condition to the proper set of conservative variables
initial_condition_transformed = transform_initial_condition(initial_condition, equations)

boundary_conditions = (; entire_boundary = boundary_condition_horizontal_slip_wall)

# A semidiscretization collects data structures and functions for the spatial discretization
semi = SemidiscretizationHyperbolic(mesh, equations, initial_condition_transformed, dg,
                                    metric_terms = metric_terms,
                                    auxiliary_field = geopotential,
                                    source_terms = source_terms_gravity_terrain_following,
                                    boundary_conditions = boundary_conditions)

###############################################################################
# ODE solvers, callbacks etc.

# Create ODE problem with time span from 0 to T
tspan = (0.0, 100.0)
ode = semidiscretize(semi, tspan)

# At the beginning of the main loop, the SummaryCallback prints a summary of the simulation
# setup and resets the timers
summary_callback = SummaryCallback()

# The AnalysisCallback allows to analyse the solution in regular intervals and prints the
# results
analysis_callback = AnalysisCallback(semi, interval = 100,
                                     save_analysis = true,
                                     extra_analysis_errors = (:conservation_error,))

# The SaveSolutionCallback allows to save the solution to a file in regular intervals
save_solution = SaveSolutionCallback(interval = 100,
                                     solution_variables = contravariant_cons2global_prim)

# The StepsizeCallback handles the re-calculation of the maximum Δt after each time step
stepsize_callback = StepsizeCallback(cfl = 0.7)

# Create a CallbackSet to collect all callbacks such that they can be passed to the ODE
# solver
callbacks = CallbackSet(summary_callback, analysis_callback, save_solution,
                        stepsize_callback)

###############################################################################
# run the simulation

# OrdinaryDiffEq's `solve` method evolves the solution in time and executes the passed
# callbacks
sol = solve(ode, CarpenterKennedy2N54(williamson_condition = false),
            dt = 0.1 , save_everystep = false, callback = callbacks)
