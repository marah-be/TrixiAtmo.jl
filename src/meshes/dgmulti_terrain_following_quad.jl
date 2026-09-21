@muladd begin
#! format: noindent

"""
    DGMultiMeshQuadTerrainFollowing2D(dg::DGMulti{2, <:Quad}, Kx, Ky, L_x, 
                                            metric_terms::MetricTermsCovariant{<:TerrainFollowingManifold};
                                            is_periodic = (true, false))
                                
Build a quadrilateral Mesh with `Kx` x `Ky` elements on a uniform grid in 
terrain-following coordinates, `[0, L_x] x [0, H_top]`, and map its nodes to
physical coordinates using the terrain-following mapping of the `TerrainFollowingManifold`
in `metric_terms`. Passing `metric_terms` ensures that the mesh and the metric terms
are based on the same orography and stretching function.

Boundary faces are not yet tagged.
"""
function DGMultiMeshQuadTerrainFollowing2D(dg::DGMulti{2, <:Quad}, Kx, Ky, L_x, 
                                            metric_terms;
                                            is_periodic = (true, false))

    (; h, H_top, f_s, f_s_inv) = metric_terms.manifold

    rd = dg.basis

    md = StartUpDG.MeshData((Kx, Ky), rd; coordinates_min=(0.0, 0.0), coordinates_max=(L_x, H_top),
        is_periodic)
    md = map_mesh_to_terrain(md, H_top, h, f_s)
    boundary_faces = StartUpDG.tag_boundary_faces(md, nothing)

    return DGMultiMesh(dg, Trixi.GeometricTermsType(Trixi.Curved(), dg), md,
                       boundary_faces)

end

# Terrain-following mapping, Baldauf (2021):
# maps a point (x1, x3) in terrain-following coordinates to physical coordinates (x1', x3')
function terrain_height_mapping(x, H_top, h, f_s)

    x1_prime = x[1]
    h_x1 = h(x[1])
    x3_prime = h_x1 + (H_top - h_x1) * f_s(x[2] / H_top)

    return SVector{2}(x1_prime, x3_prime)

end

# Apply the terrain-following mapping to the nodes of 
# a uniform mesh in terrain-following coordinates
function map_mesh_to_terrain(md, H_top, h, f_s)

    x1, x3 = md.xyz
    nodes_tf = SVector.(x1, x3)
    nodes_prime = terrain_height_mapping.(nodes_tf, H_top, h, f_s)
    x3_prime = getindex.(nodes_prime, 2)
    xyz = (x1, x3_prime)

    x1q, x3q = md.xyzq
    nodes_tf_q = SVector.(x1q, x3q)
    nodes_prime_q = terrain_height_mapping.(nodes_tf_q, H_top, h, f_s)
    x3q_prime = getindex.(nodes_prime_q, 2)
    xyzq = (x1q, x3q_prime)

    x1f, x3f = md.xyzf
    nodes_tf_f = SVector.(x1f, x3f)
    nodes_prime_f = terrain_height_mapping.(nodes_tf_f, H_top, h, f_s)
    x3f_prime = getindex.(nodes_prime_f, 2)
    xyzf = (x1f, x3f_prime)

    md = setproperties(md, xyz=xyz, xyzq=xyzq, xyzf=xyzf)

    return md

end

end # @muladd