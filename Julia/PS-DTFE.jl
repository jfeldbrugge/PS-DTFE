using LinearAlgebra, StaticArrays, Delaunay

struct BVH
    data::Union{Vector{Int}, Nothing}
    box::Matrix{Float64}
    depth::Int
    left::Union{BVH, Nothing}
    right::Union{BVH, Nothing}
end

function BVH(data, box::Matrix{Float64}, points::Matrix{Float64}, simplices, depth::Int)
    if depth != 0
        dim = (depth % size(box, 1)) + 1
        @inbounds mins = vec(minimum(@view(points[@view(simplices[data, :]), dim]), dims = 2))
        @inbounds maxs = vec(maximum(@view(points[@view(simplices[data, :]), dim]), dims = 2))
        
        @inbounds div = sum(@view(box[dim,:])) / length(@view(box[dim,:]))
        L = mins .<= div
        R = maxs .>= div

        Lbox = copy(box)
        @inbounds Lbox[dim,:] = [Lbox[dim, 1], div]
        Rbox = copy(box)
        @inbounds Rbox[dim,:] = [div, Rbox[dim, 2]]

        @inbounds BVH(nothing, box, depth, BVH(data[L], Lbox, points, simplices, depth - 1), 
                                           BVH(data[R], Rbox, points, simplices, depth - 1))
    else 
        BVH(data, box, depth, nothing, nothing)
    end
end

function findCandidateSimplices(p::Vector{Float64}, BVH_tree::BVH)
    dim = (BVH_tree.depth % size(BVH_tree.box, 1)) + 1
    if BVH_tree.depth == 0
        return BVH_tree.data
    elseif p[dim] < BVH_tree.left.box[dim, 2]
        return findCandidateSimplices(p, BVH_tree.left)
    elseif p[dim] > BVH_tree.left.box[dim, 2]
        return findCandidateSimplices(p, BVH_tree.right)
    else
        return union(findCandidateSimplices(p, BVH_tree.left),
                     findCandidateSimplices(p, BVH_tree.right))
    end
end

function findBox(p::Vector{Float64}, BVH_tree::BVH)
    dim = (BVH_tree.depth % size(BVH_tree.box, 1)) + 1
    if BVH_tree.depth == 0
        println(BVH_tree.box)
    elseif p[dim] < BVH_tree.left.box[dim, 2]
        return findBox(p, BVH_tree.left)
    elseif p[dim] > BVH_tree.left.box[dim, 2]
        return findBox(p, BVH_tree.right)
    else
        findBox(p, BVH_tree.left)
        findBox(p, BVH_tree.right)
    end
end

function intersection(p::Vector{Float64}, simplex)
    @inbounds barry = inv(@view(simplex[2:end,:])' .- @view(simplex[1,:])) * (p .- @view(simplex[1,:]))
    return all(barry .>= 0) & all(barry .<= 1) & (sum(barry) <= 1.)
end

function findIntersections(p::Vector{Float64}, BVH_tree::BVH, points, simplices)
    candidates = findCandidateSimplices(p, BVH_tree)
    @inbounds filter(i -> intersection(p, @view(points[@view(simplices[i,:]),:])), candidates)
end

volume(sim, points) = @inbounds abs(det(@view(points[@view(sim[2:end]),:])' .- @view(points[sim[1],:]))) /  factorial(size(points, 2))

struct PS_DTFE
    rho::Vector{Float64}
    Drho::Matrix{Float64}
    Dv::Array{Float64}
    tree::BVH
    simplices::Matrix{Int}
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
    positions_initial::Matrix{Float64}

    function PS_DTFE(positions_initial, positions, velocities, m, depth, box)


        positions = unwrap_x_(positions_initial, positions, L);
        positions_initial, positions = translate(positions_initial, positions, L)
        velocities          = frame_velocities(positions, velocities, L)
        positions_initial, positions = frame(positions_initial, positions, L)

        simplices = delaunay(positions_initial).simplices

        dim = size(box, 1)
    
        BVH_tree = BVH(1:size(simplices, 1), box, positions, simplices, depth * dim)
    
        rho = zeros(size(positions,1))
        @inbounds for i in axes(simplices, 1)
            vol = volume(@view(simplices[i,:]), positions)
            @inbounds for index in @view(simplices[i,:])
                rho[index] += vol
            end
        end
        rho = (1. + dim) * m ./ rho
    
        Drho = zeros(size(simplices, 1), dim)
        Dv   = zeros(size(simplices, 1), dim, dim)
    
        @inbounds for i in axes(simplices, 1)
            sim = @view(simplices[i,:])
            p = @view(positions[sim,:])
            r = @view(rho[sim,:])
            v = @view(velocities[sim,:])
            A_inv = inv(@view(p[2:end,:])' .- @view(p[1,:]))'
            Drho[i,:] = A_inv * (@view(r[2:end]) .- r[1])
            Dv[i,:,:] = A_inv * ((@view(v[2:end,:])' .- @view(v[1,:]))')
        end
    
        return new(rho, Drho, Dv, BVH_tree, simplices, positions, velocities, positions_initial)
    end
end

function density(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)

    dens = 0.
    @inbounds for simplexIndex in simplexIndices
        pointIndex = estimator.simplices[simplexIndex,1]
        dens += estimator.rho[pointIndex] + @view(estimator.Drho[simplexIndex,:])' * (p .- @view(estimator.positions[pointIndex,:]))
    end
    return dens
end

function v(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)
    println("simplex_indices = ", simplexIndices)

    vs = zeros(length(simplexIndices), length(p))

    @inbounds for (i, simplexIndex) in pairs(simplexIndices)
        pointIndex = estimator.simplices[simplexIndex, 1]
        vs[i,:] = @view(estimator.velocities[pointIndex,:]) + @view(estimator.Dv[simplexIndex,:,:]) * (p .- @view(estimator.positions[pointIndex,:]))
    end
    return vs
end

function numberOfStreams(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)
    return length(simplexIndices)
end


function vSum(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)

    vs = zeros(length(simplexIndices), length(p))

    @inbounds for (i, simplexIndex) in pairs(simplexIndices)
        pointIndex = estimator.simplices[simplexIndex, 1]
        vs[i,:] = @view(estimator.velocities[pointIndex,:]) + @view(estimator.Dv[simplexIndex,:,:]) * (p .- @view(estimator.positions[pointIndex,:]))
    end

    if size(vs) == (1,3)  # single-stream region
        return vs
    else   # multistream region
        densities = zeros(length(simplexIndices))
        for (i, simplexIndex) in pairs(simplexIndices)
            pointIndex   = estimator.simplices[simplexIndex,1]
            densities[i] = estimator.rho[pointIndex] + @view(estimator.Drho[simplexIndex,:])' * (p .- @view(estimator.positions[pointIndex,:]))
        end

        #stream-density weighted sum of velocities
        return sum(densities .* vs, dims=1) / sum(densities)
    end
end

"""
    unwrap_x_(q, x, L)

Cosmological N-body simulations normally work with periodic boundary conditions. This function undoes the wrapping of particles around the box, assuming the particles did not travel more than half the size of the box.
"""
function unwrap_x_(q, x, L)
    unwrap_s.(x - q, L) + q
end

"""
    unwrap_s(s, L)

Cosmological N-body simulations normally work with periodic boundary conditions. This function undoes the wrapping of particles around the box in the displacement field, assuming the particles did not travel more than half the size of the box.
"""
unwrap_s(s, L) = mod((s + L / 2), L) - L / 2

"""
    translate(coords_q, coords_x, L)

Cosmological N-body simulations normally work with periodic boundary conditions. This function shifts both the initial and final positions of the particles such that they are located in the simulation box in Eulerian space.
"""
function translate(coords_q, coords_x, L)
    coords_q_ = copy(coords_q)
    coords_x_ = copy(coords_x)

    let index_x = coords_x .< 0.
        coords_q_[index_x] .+= L
        coords_x_[index_x] .+= L
    end
        
    let index_x = coords_x .> L
        coords_q_[index_x] .-= L
        coords_x_[index_x] .-= L
    end
    return coords_q_, coords_x_
end

"""
    frame(coords_q, coords_x, L, pad=0.05)

Cosmological N-body simulations normally work with periodic boundary conditions. This function adds a frame of periodic particle positions around the simulation box to implement periodicity in the Delaunay tesselation.
`pad` specifies the width of the frame in units of the simulation box size `L`.
"""
function frame(coords_q, coords_x, L, pad=0.05)
    for d in 1:size(coords_q,2)
        indexL = coords_x[:,d] .> (1. - pad) * L
        indexR = coords_x[:,d] .< pad * L

        newPointsL_q = coords_q[indexL, :]
        newPointsL_x = coords_x[indexL, :]
        newPointsL_q[:,d] .-= L 
        newPointsL_x[:,d] .-= L 
        
        newPointsR_q = coords_q[indexR, :]
        newPointsR_x = coords_x[indexR, :]
        newPointsR_q[:,d] .+= L 
        newPointsR_x[:,d] .+= L 

        coords_q = vcat(coords_q, newPointsL_q)        
        coords_q = vcat(coords_q, newPointsR_q)
        coords_x = vcat(coords_x, newPointsL_x)        
        coords_x = vcat(coords_x, newPointsR_x)
    end

    return coords_q, coords_x
end

"""
    frame_velocities(coords_x, velocities, L, pad=0.05)

Cosmological N-body simulations normally work with periodic boundary conditions. This function adds a frame of velocities (corresponding to the periodic particle positions) around the simulation box to implement periodicity in the Delaunay tesselation.
`pad` specifies the width of the frame in units of the simulation box size `L`.
"""
function frame_velocities(coords_x, velocities, L, pad=0.05)
    for d in 1:size(coords_x,2)
        indexL = coords_x[:,d] .> (1. - pad) * L
        indexR = coords_x[:,d] .< pad * L

        newVelocitiesL = velocities[indexL, :]
        newPointsL_x   = coords_x[indexL, :]
        newPointsL_x[:,d] .-= L 
        
        newVelocitiesR = velocities[indexR, :]
        newPointsR_x   = coords_x[indexR, :]
        newPointsR_x[:,d] .+= L 

        coords_x = vcat(coords_x, newPointsL_x)        
        coords_x = vcat(coords_x, newPointsR_x)
        velocities = vcat(velocities,  newVelocitiesL)        
        velocities = vcat(velocities,  newVelocitiesR)        
    end

    return velocities
end
