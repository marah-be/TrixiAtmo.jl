@muladd begin
#! format: noindent 

function init_auxiliary_node_variables!(aux_values, mesh::DGMultiMesh,
                                        equations::AbstractCovariantEquations{NDIMS,
                                                                              NDIMS_AMBIENT},
                                        dg::DGMulti{NDIMS},
                                        metric_terms,
                                        auxiliary_field) where {NDIMS, NDIMS_AMBIENT}
    rd = dg.basis
    (; xyz) = mesh.md
    n_aux = n_aux_node_vars(equations)

    # Identify the vertices corresponding to the corners of the reference element. rd.V1 is not useful,
    # since the physical nodes are projected onto the sphere and thus the corner nodes would be slightly off.
    # Instead, we identify the corner nodes by their reference coordinates and build a mask to access them directly
    VMask = compute_vertex_mask(dg)

    # Compute the radius of the sphere from the first element's fourth vertex, such that we can use it
    # throughout the computation. We assume that each Wedge element's last three corner vertices lie
    # on the simulated sphere.
    VXYZ = map(coords -> coords[VMask, 1], xyz)
    v_outer = getindex.(VXYZ, 1)
    radius = norm(v_outer)

    for element in eachelement(mesh, dg)
        # Compute corner vertices of the element
        VXYZ = map(coords -> coords[VMask, element], xyz)
        vertices = map(i -> SVector(getindex.(VXYZ, i)), 1:length(VXYZ[1]))

        aux_node = Vector{eltype(aux_values[1, 1])}(undef, n_aux)

        # Compute the auxiliary metric information at each node
        for i in 1:Trixi.nnodes(dg)
            # Physical coordinates of the node
            x_node = map(coords -> coords[i, element], xyz)

            # Covariant basis in the desired global coordinate system as columns of a matrix
            basis_covariant = calc_basis_covariant(vertices,
                                                   getindex.(rd.rst[1:NDIMS], i)...,
                                                   radius, dg, metric_terms,
                                                   equations.global_coordinate_system)

            calc_aux_node!(aux_node, basis_covariant, x_node, equations,
                           auxiliary_field)

            aux_values[i, element] = SVector{n_aux}(aux_node)
        end
        # Christoffel symbols of the second kind fill up the rest of vector
        calc_christoffel_symbols!(aux_values, mesh, equations, metric_terms, dg,
                                  element, vertices, radius)
    end

    return nothing
end

function calc_aux_node!(aux_node, basis_covariant, x_node,
                        equations::AbstractCovariantEquations{2, 2},
                        geopotential)
    aux_node[1:4] = SVector(basis_covariant)

    # Covariant metric tensor G := basis_covariant' * basis_covariant
    metric_covariant = basis_covariant' * basis_covariant

    # Contravariant metric tensor inv(G)
    metric_contravariant = inv(metric_covariant)

    # Contravariant basis vectors as rows of a matrix
    basis_contravariant = metric_contravariant * basis_covariant'
    aux_node[5:8] = SVector(basis_contravariant)

    # Area element
    aux_node[9] = sqrt(det(metric_covariant))

    # Covariant metric tensor components
    aux_node[10:12] = SVector(metric_covariant[1, 1],
                              metric_covariant[1, 2],
                              metric_covariant[2, 2])

    # Contravariant metric tensor components
    aux_node[13:15] = SVector(metric_contravariant[1, 1],
                              metric_contravariant[1, 2],
                              metric_contravariant[2, 2])

    # Geopotential
    if !isnothing(geopotential)
        aux_node[16] = geopotential(x_node, equations)
    else
        aux_node[16] = zero(eltype(aux_node))
    end
end

function calc_aux_node!(aux_node, basis_covariant, x_node,
                        equations::AbstractCovariantEquations{2, 3},
                        bottom_topography)
    aux_node[1:6] = SVector(basis_covariant)

    # Covariant metric tensor G := basis_covariant' * basis_covariant
    metric_covariant = basis_covariant' * basis_covariant

    # Contravariant metric tensor inv(G)
    metric_contravariant = inv(metric_covariant)

    # Contravariant basis vectors as rows of a matrix
    basis_contravariant = metric_contravariant * basis_covariant'
    aux_node[7:12] = SVector(basis_contravariant)

    # Area element
    aux_node[13] = sqrt(det(metric_covariant))

    # Covariant metric tensor components
    aux_node[14:16] = SVector(metric_covariant[1, 1],
                              metric_covariant[1, 2],
                              metric_covariant[2, 2])

    # Contravariant metric tensor components
    aux_node[17:19] = SVector(metric_contravariant[1, 1],
                              metric_contravariant[1, 2],
                              metric_contravariant[2, 2])

    # Bottom topography
    if !isnothing(bottom_topography)
        aux_node[20] = bottom_topography(x_node, equations)
    else
        aux_node[20] = zero(eltype(aux_node))
    end
end

function compute_vertex_mask(dg::DGMulti{<:Any, <:Tri})
    rd = dg.basis
    VMask = []
    for corner in [(-1.0, -1.0), (1.0, -1.0), (-1.0, 1.0)]
        for j in 1:size(rd.rst[1], 1)
            r, s = rd.rst[1][j], rd.rst[2][j]
            if all(isapprox.((r, s), corner))
                push!(VMask, j)
            end
        end
    end
    return VMask
end

function compute_vertex_mask(dg::DGMulti{<:Any, <:Quad})
    rd = dg.basis
    VMask = []
    for corner in [(-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0)]
        for j in 1:size(rd.rst[1], 1)
            r, s = rd.rst[1][j], rd.rst[2][j]
            if all(isapprox.((r, s), corner))
                push!(VMask, j)
            end
        end
    end
    return VMask
end

# Analytically compute the transformation matrix A, such that G = AᵀA is the
# covariant metric tensor and a_i = A[1,i] * e_x + A[2,i] * e_y + A[3,i] * e_z denotes
# the covariant tangent basis, where e_x, e_y, and e_z are the Cartesian unit basis vectors.
@inline function calc_basis_covariant(vertices, r, s, radius, dg::DGMulti{2, <:Tri},
                                      ::MetricTermsCovariant{SphericalManifold},
                                      ::GlobalCartesianCoordinates)
    v1, v2, v3 = vertices

    # Construct a bilinear mapping based on the three corner vertices
    xe = 0.5f0 * (-(r + s) * v1 + (1 + r) * v2 +
                  (1 + s) * v3)
    # Derivatives of bilinear map with respect to reference coordinates xi1, xi2
    dxedr = 0.5f0 *
            (-v1 + v2)
    dxeds = 0.5f0 *
            (-v1 + v3)

    # Use product/quotient rule on the projection
    norm_xe = norm(xe)
    dxdr = radius / norm_xe * (dxedr - dot(xe, dxedr) / norm_xe^2 * xe)
    dxds = radius / norm_xe * (dxeds - dot(xe, dxeds) / norm_xe^2 * xe)

    return SMatrix{3, 2}(dxdr[1], dxdr[2], dxdr[3],
                         dxds[1], dxds[2], dxds[3])
end

@inline function calc_basis_covariant(vertices, r, s, radius, dg::DGMulti{2, <:Quad},
                                      ::MetricTermsCovariant{FlatManifold},
                                      ::GlobalCartesianCoordinates)
    v1, v2, v3, v4 = vertices

    # Construct a bilinear mapping based on the three corner vertices
    xe = 0.25f0 * (((1 - r) * (1 - s) * v1 + (1 + r) * (1 - s) * v2 +
           (1 + r) * (1 + s) * v3 + (1 - r) * (1 + s) * v4))
    # Derivatives of bilinear map with respect to reference coordinates xi1, xi2
    dxedr = 0.25f0 *
            (((-(1 - s) * v1 + (1 - s) * v2 + (1 + s) * v3 - (1 + s) * v4)))
    dxeds = 0.25f0 *
            (((-(1 - r) * v1 - (1 + r) * v2 + (1 + r) * v3 + (1 - r) * v4)))

    return SMatrix{2, 2}(dxedr[1], dxedr[2],
                         dxeds[1], dxeds[2])
end

# Inverse of the quadratic stretching function f_s(y) = beta * y^2 + (1 - beta) * y
# from Baldauf (2021), Sec. 5.2
function stretching_function_inverse(phi, beta)

    if beta == 0
        return phi
    end

    return (-(1-beta) + sqrt((1-beta)^2 + (4 * beta * phi))) / (2 * beta)

end

# Inverse of the terrain-follwing mapping, Baldauf (2021), Eq. (57): 
# maps a point in physical coodinates back to terrain-following coordinates
# Requires h(x1) < H_top and a strictly monotonic stretching function
function terrain_height_mapping_inverse(x_prime, H_top, h, f_s_inv)

    x1 = x_prime[1]
    phi = (x_prime[2] - h(x1)) / (H_top - h(x1))

    x3 = f_s_inv(phi) * H_top

    return SVector{2}(x1, x3)

end

# Jacobian of the terrain-following mapping, Baldauf (2021), Eq. (57),
# with respect to the terrain_followng coordinates (x1, x3) 
function terrain_height_jacobian(x1, x3, H_top, h, f_s)

    dx1dx1 = 1.0
    dx1dx3 = 0.0
    dx3dx1 = derivative(h, x1) * (1 - f_s(x3/H_top))
    dx3dx3 = (H_top - h(x1)) * derivative(f_s, (x3/H_top)) * (1/H_top)

    return SMatrix{2, 2}(dx1dx1, dx3dx1,
                         dx1dx3, dx3dx3)

end


@inline function calc_basis_covariant(vertices, r, s, radius, dg::DGMulti{2, <:Quad}, 
                                        metric_terms::MetricTermsCovariant{<:TerrainFollowingManifold},
                                        ::GlobalCartesianCoordinates)

    v1, v2, v3, v4 = vertices
    (; h, H_top, f_s, f_s_inv) = metric_terms.manifold

    # Horizontal element boundaries (unchanged by the mapping)
    x1_a = v1[1]
    x1_b = v2[1]

    # Vertical element boundaries in terrain-following coordinates
    # recovered from the physical vertices via the inverse mapping
    v1_tf = terrain_height_mapping_inverse(v1, H_top, h, f_s_inv)
    v4_tf = terrain_height_mapping_inverse(v4, H_top, h, f_s_inv)
    x3_a = v1_tf[2]
    x3_b = v4_tf[2]

    # Affine map from reference coordinates (r, s) to terrain-following coordinates
    x1 = (r + 1) * (x1_b - x1_a)  / 2 + x1_a
    x3 = (s + 1)* (x3_b - x3_a) / 2 + x3_a

    # Apply the chain rule to the composite mapping (r, s) -> (x1, x3) -> (x1', x3')
    return terrain_height_jacobian(x1, x3, H_top, h, f_s) * SMatrix{2, 2}((x1_b - x1_a) / 2,  0.0,
                                                                           0.0, (x3_b - x3_a) / 2)
end

# Calculate the covariant metric tensor components G₁₁, G₁₂ (= G₂₁), and G₂₂ and return in
# that order as an SVector of length 3
function calc_metric_covariant(vertices, xi1, xi2, radius, dg, metric_terms,
                               equations::AbstractCovariantEquations{2})
    A = calc_basis_covariant(vertices, xi1, xi2, radius, dg, metric_terms,
                             equations.global_coordinate_system)
    Gcov = A' * A
    return SVector(Gcov[1, 1], Gcov[1, 2], Gcov[2, 2])
end

# Use ForwardDiff.jl to automatically differentiate the covariant metric tensor components
function calc_metric_derivatives_autodiff(vertices, xi1, xi2, radius, dg, metric_terms,
                                          equations::AbstractCovariantEquations{2})
    dGdxi1 = derivative(x -> calc_metric_covariant(vertices, x, xi2, radius, dg,
                                                   metric_terms, equations), xi1)
    dGdxi2 = derivative(x -> calc_metric_covariant(vertices, xi1, x, radius, dg,
                                                   metric_terms, equations), xi2)
    return dGdxi1, dGdxi2
end

# Use the collocation derivative operator to numerically differentiate the covariant
# metric tensor components
function calc_metric_derivatives_collocation(aux_values,
                                             equations::AbstractCovariantEquations{2},
                                             dg::DGMulti, i, element)
    rd = dg.basis
    Dr, Ds = rd.Drst

    # Numerically differentiate covariant metric components with respect to ξ¹
    dG11dxi1 = zero(eltype(aux_values[i, element]))
    dG12dxi1 = zero(eltype(aux_values[i, element]))
    dG22dxi1 = zero(eltype(aux_values[i, element]))
    for jj in 1:nnodes(dg)
        aux_node_jj = aux_values[jj, element]
        Gcov_jj = metric_covariant(aux_node_jj, equations)
        dG11dxi1 += Dr[i, jj] * Gcov_jj[1, 1]
        dG12dxi1 += Dr[i, jj] * Gcov_jj[1, 2]
        dG22dxi1 += Dr[i, jj] * Gcov_jj[2, 2]
    end

    # Numerically differentiate covariant metric components with respect to ξ²
    dG11dxi2 = zero(eltype(aux_values[i, element]))
    dG12dxi2 = zero(eltype(aux_values[i, element]))
    dG22dxi2 = zero(eltype(aux_values[i, element]))
    for jj in 1:nnodes(dg)
        aux_node_jj = aux_values[jj, element]
        Gcov_jj = metric_covariant(aux_node_jj, equations)
        dG11dxi2 += Ds[i, jj] * Gcov_jj[1, 1]
        dG12dxi2 += Ds[i, jj] * Gcov_jj[1, 2]
        dG22dxi2 += Ds[i, jj] * Gcov_jj[2, 2]
    end

    return SVector(dG11dxi1, dG12dxi1, dG22dxi1), SVector(dG11dxi2, dG12dxi2, dG22dxi2)
end

function calc_christoffel_symbols!(aux_values, mesh::DGMultiMesh,
                                   equations::AbstractCovariantEquations{2},
                                   metric_terms::MetricTermsCovariant{<:Any,
                                                                      ChristoffelSymbolsAutodiff},
                                   dg, element, vertices, radius)
    rd = dg.basis
    for i in 1:Trixi.nnodes(dg)
        # Compute metric derivatives using automatic differentiation
        dGdxi1, dGdxi2 = calc_metric_derivatives_autodiff(vertices, rd.rst[1][i],
                                                          rd.rst[2][i], radius,
                                                          dg, metric_terms, equations)

        aux_node = Vector(aux_values[i, element])
        Gcon = metric_contravariant(aux_node, equations)
        aux_node[(end - 5):end] .= calc_christoffel_symbols(dGdxi1, dGdxi2, Gcon)

        aux_values[i, element] = SVector{n_aux_node_vars(equations)}(aux_node)
    end
end

function calc_christoffel_symbols!(aux_values, mesh::DGMultiMesh,
                                   equations::AbstractCovariantEquations{2},
                                   metric_terms::MetricTermsCovariant{<:Any,
                                                                      ChristoffelSymbolsCollocationDerivative},
                                   dg, element, vertices, radius)
    for i in 1:Trixi.nnodes(dg)
        # Compute metric derivatives using collocation differentiation
        dGdxi1, dGdxi2 = calc_metric_derivatives_collocation(aux_values, equations, dg,
                                                             i, element)

        aux_node = Vector(aux_values[i, element])
        Gcon = metric_contravariant(aux_node, equations)
        aux_node[(end - 5):end] .= calc_christoffel_symbols(dGdxi1, dGdxi2, Gcon)

        aux_values[i, element] = SVector{n_aux_node_vars(equations)}(aux_node)
    end
end
end # @muladd
