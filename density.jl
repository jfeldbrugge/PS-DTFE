using LinearAlgebra
using Delaunay

struct BVH
    data::Union{Vector{Int}, Nothing}
    box::Matrix{Float64}
    depth::Int
    left::Union{BVH, Nothing}
    right::Union{BVH, Nothing}
end

BVH_constructor = function(data, box, points, simplices, depth)
    if depth != 0
        dim = (depth % size(box)[1]) + 1
        mins = vec(minimum(positions[simplices[data, :], dim], dims = 2))
        maxs = vec(maximum(positions[simplices[data, :], dim], dims = 2))
        
        mean(x) = sum(x) / length(x)

        div = mean(box[dim,:])
        L = mins .<= div
        R = maxs .>= div

        Lbox = copy(box)
        Lbox[dim,:] = [Lbox[dim, 1], div]
        Rbox = copy(box)
        Rbox[dim,:] = [div, Rbox[dim, 2]]

        BVH(nothing, box, depth, BVH_constructor(data[L], Lbox, points, simplices, depth - 1), 
                                 BVH_constructor(data[R], Rbox, points, simplices, depth - 1))
    else 
        BVH(data, box, depth, nothing, nothing)
    end
end

findCandidateSimplices = function (p::Vector{Float64}, BVH_tree::BVH)
    dim = (BVH_tree.depth % size(BVH_tree.box)[1]) + 1
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

findBox = function (p::Vector{Float64}, BVH_tree::BVH)
    dim = (BVH_tree.depth % size(BVH_tree.box)[1]) + 1
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

intersection = function (p::Vector{Float64}, simplex::Matrix{Float64})
    barry = inv(simplex[2:end,:]' .- simplex[1,:]) * (p .- simplex[1,:])
    return all(barry .>= 0) & all(barry .<= 1) & (sum(barry) <= 1.)
end

findIntersections = function (p::Vector{Float64}, BVH_tree::BVH, points, simplices)
    candidates = findCandidateSimplices(p, BVH_tree)
    filter(i -> intersection(p, points[simplices[i,:],:]), candidates)
end

volume(sim, points) = abs(det(points[sim[2:end],:]' .- points[sim[1],:])) /  factorial(size(points)[2])

struct ps_dtfe
    rho::Vector{Float64}
    Drho::Matrix{Float64}
    Dv::Array{Float64}
    tree::BVH
    simplices::Matrix{Int}
    positions::Matrix{Float64}
    velocities::Matrix{Float64}
end

ps_dtfe_constructor = function (positions_initial, positions, velocities, m, depth, box)
    simplices = delaunay(positions_initial).simplices
    dim = size(box)[1]

    BVH_tree = BVH_constructor(1:size(simplices)[1], box, positions, simplices, depth * dim)

    rho = zeros(size(positions)[1])
    for i in axes(simplices, 1)
        vol = volume(simplices[i,:], positions)
        for index in simplices[i,:]
            rho[index] += vol
        end
    end
    rho = (1. + dim) * m ./ rho

    Drho = zeros(size(simplices)[1], dim)
    Dv   = zeros(size(simplices)[1], dim, dim)

    for i in axes(simplices, 1)
        p = positions[simplices[i,:],:]
        r = rho[simplices[i,:],:]
        v = velocities[simplices[i,:],:]
        A_inv = inv(p[2:end,:]' .- p[1,:])'
        Drho[i,:] = A_inv * (r[2:end] .- r[1])
        Dv[i,:,:] = A_inv * ((v[2:end,:]' .- v[1,:])')
    end

    return ps_dtfe(rho, Drho, Dv, BVH_tree, simplices, positions, velocities)
end

density = function (p::Vector{Float64}, estimator::ps_dtfe)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)

    dens = 0.
    for simplexIndex in simplexIndices
        pointIndex = estimator.simplices[simplexIndex,1]
        dens += estimator.rho[pointIndex] + estimator.Drho[simplexIndex,:]' * (p .- estimator.positions[pointIndex,:])
    end
    return dens
end

v = function (p::Vector{Float64}, estimator::ps_dtfe)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)

    vs = zeros(length(simplexIndices), length(p))

    for (i, simplexIndex) in pairs(simplexIndices)
        pointIndex = estimator.simplices[simplexIndex, 1]
        vs[i,:] = estimator.velocities[pointIndex,:] + estimator.Dv[simplexIndex,:,:] * (p .- estimator.positions[pointIndex,:])
    end
    return vs
    
end

numberOfStreams = function (p::Vector{Float64}, estimator::ps_dtfe)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)
    return length(simplexIndices)
end