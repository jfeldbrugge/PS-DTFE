using LinearAlgebra, StaticArrays, TetGen, Serialization, ProgressMeter

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
        
        # obtain tesselation efficiently through custom TetGen use
        inputTetGen = TetGen.RawTetGenIO{Float64}()
        inputTetGen.pointlist = copy(positions_initial')
        simplices = Int64.(tetrahedralize(inputTetGen,"Q").tetrahedronlist')
        inputTetGen = nothing
        
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

function numberOfStreams(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)
    return length(simplexIndices)
end

function velocity(p::Vector{Float64}, estimator::PS_DTFE, single_stream=false)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)

    vs = zeros(length(simplexIndices), length(p))

    @inbounds for (i, simplexIndex) in pairs(simplexIndices)
        pointIndex = estimator.simplices[simplexIndex, 1]
        vs[i,:] = @view(estimator.velocities[pointIndex,:]) + @view(estimator.Dv[simplexIndex,:,:]) * (p .- @view(estimator.positions[pointIndex,:]))
    end

    if single_stream && size(vs) != (1,3)  # in multistream region
        return [NaN NaN NaN]
    else
        return vs
    end
end

function velocitySum(p::Vector{Float64}, estimator::PS_DTFE)
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

function inSimplices(p::Vector{Float64}, estimator::PS_DTFE)
    simplexIndices = findIntersections(p, estimator.tree, estimator.positions, estimator.simplices)
end



###############################################################################
## Subbox PS-DTFE calculation
###############################################################################

## auxiliary functions —-------------------------------------------------------

function unwrap_x_(q, x, L)
    unwrap_s.(x - q, L) + q
end

unwrap_s(s, L) = mod((s + L / 2), L) - L / 2

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


function frame(coords_q, coords_x, L, pad=0.05)
    for d in 1:3
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

function frame_velocities(coords_x, velocities, L, pad=0.05)
    for d in 1:3
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


## subbox PS-DTFE routines ----------------------------------------------------

struct PS_DTFE_subbox
    N_sub::Int64
    N_target::Int64
    m::Float64
    depth::Int64
    dir::String
    L::Float64
    Ni::Int64
end

struct SimBox
    L::Float64
    Ni::Int64
end

function ps_dtfe_subbox(coords_q, coords_x, m, depth, sim_box::SimBox; N_target=128, pad=0.05, dir="./ps_dtfe")

    mkpath(dir)  # succeeds even if already exists
    rm(dir, force=true, recursive=true)
    mkdir(dir)

    # subboxes per side length
    N_sub = sim_box.Ni ÷ N_target
    println("Calculating $(N_sub^3) subbox estimators with $N_target^3 particles each (total simulation $(sim_box.Ni)^3 particles).")

    # pre-process coordinates: unwrapping, translating and padding
    coords_x = unwrap_x_(coords_q, coords_x, sim_box.L);
    coords_q_, coords_x_ = translate(coords_q, coords_x, sim_box.L)
    coords_q_, coords_x_ = frame(coords_q_, coords_x_, sim_box.L, pad);

    # iterate over subboxes and create subbox ps_dtfe
    idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]
    @showprogress for (i, j, k) in idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        get_subbox_estimator(coords_q_, coords_x_, [i, j, k], N_sub, m, depth, sim_box; pad=pad, dir=dir)
    end

    return PS_DTFE_subbox(N_sub, N_target, m, depth, dir, L, sim_box.Ni)
end

function ps_dtfe_subbox(coords_q, coords_x, velocities, m, depth, sim_box::SimBox; N_target=128, pad=0.05, dir="./ps_dtfe")

    mkpath(dir)  # succeeds even if already exists
    rm(dir, force=true, recursive=true)
    mkdir(dir)

    # subboxes per side length
    N_sub = sim_box.Ni ÷ N_target
    println("Calculating $(N_sub^3) subbox estimators with $N_target^3 particles each (total simulation $(sim_box.Ni)^3 particles).")

    # pre-process coordinates: unwrapping, translating and padding
    coords_x = unwrap_x_(coords_q, coords_x, sim_box.L);
    coords_q_, coords_x_ = translate(coords_q, coords_x, sim_box.L)
    velocities_          = frame_velocities(coords_x_, velocities, sim_box.L, pad)
    coords_q_, coords_x_ = frame(coords_q_, coords_x_, sim_box.L, pad);

    # iterate over subboxes and create subbox ps_dtfe
    idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]
    @showprogress for (i, j, k) in idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        get_subbox_estimator(coords_q_, coords_x_, velocities_, [i, j, k], N_sub, m, depth, sim_box; pad=pad, dir=dir)
    end

    return PS_DTFE_subbox(N_sub, N_target, m, depth, dir, L, sim_box.Ni)
end

function get_subbox_estimator(coords_q, coords_x, idx, N_sub, m, depth, sim_box::SimBox; pad=0.05, dir="./ps_dtfe")
    box = 1. / N_sub * sim_box.L .* hcat(idx, idx .+ 1)
    
    idx_x  = (box[1,1] - pad*sim_box.L .< coords_x[:,1] .< box[1,2] + pad*sim_box.L) .& 
             (box[2,1] - pad*sim_box.L .< coords_x[:,2] .< box[2,2] + pad*sim_box.L) .& 
             (box[3,1] - pad*sim_box.L .< coords_x[:,3] .< box[3,2] + pad*sim_box.L) 
    Ni_ = size(coords_q[idx_x, :])[1]

    ps_dtfe = PS_DTFE(coords_q[idx_x, :], coords_x[idx_x, :], zeros(Ni_, 3), m, depth, box)
    i, j, k = idx
    serialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k), ps_dtfe)
end

function get_subbox_estimator(coords_q, coords_x, velocities, idx, N_sub, m, depth, sim_box::SimBox; pad=0.05, dir="./ps_dtfe")
    box = 1. / N_sub * sim_box.L .* hcat(idx, idx .+ 1)
    
    idx_x  = (box[1,1] - pad*sim_box.L .< coords_x[:,1] .< box[1,2] + pad*sim_box.L) .& 
             (box[2,1] - pad*sim_box.L .< coords_x[:,2] .< box[2,2] + pad*sim_box.L) .& 
             (box[3,1] - pad*sim_box.L .< coords_x[:,3] .< box[3,2] + pad*sim_box.L) 
    Ni_ = size(coords_q[idx_x, :])[1]

    ps_dtfe = PS_DTFE(coords_q[idx_x, :], coords_x[idx_x, :], velocities[idx_x, :], m, depth, box)
    i, j, k = idx
    serialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k), ps_dtfe)
end


function get_coords_in_subbox(coords, idx, N_sub, L)
    box = 1. / N_sub * L .* hcat(idx, idx .+ 1)

    coord_indices  = CartesianIndices(coords)
    in_subbox      = CartesianIndex[]

    for coord_idx in coord_indices
        p = coords[coord_idx]

        ## if box has index such that it contains a "right-hand boundary" (x = L or y = L or z = L),
        ## need to consider for subbox range

        if all(idx .<  N_sub-1) ## not at "right-hand boundary"
            if (box[1,1] <= p[1] < box[1,2]) & (box[2,1] <= p[2] < box[2,2]) & (box[3,1] <= p[3] < box[3,2])
                push!(in_subbox, coord_idx)
            end

        else  ## at least one "right-hand boundary" -> distinguish cases

            if (idx[1] == N_sub-1) && !(idx[2] == N_sub-1) && !(idx[3] == N_sub-1)
                if (box[1,1] <= p[1] <= box[1,2]) & (box[2,1] <= p[2] < box[2,2]) & (box[3,1] <= p[3] < box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif !(idx[1] == N_sub-1) && (idx[2] == N_sub-1) && !(idx[3] == N_sub-1)
                if (box[1,1] <= p[1] < box[1,2]) & (box[2,1] <= p[2] <= box[2,2]) & (box[3,1] <= p[3] < box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif !(idx[1] == N_sub-1) && !(idx[2] == N_sub-1) && (idx[3] == N_sub-1)
                if (box[1,1] <= p[1] < box[1,2]) & (box[2,1] <= p[2] < box[2,2]) & (box[3,1] <= p[3] <= box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif (idx[1] == N_sub-1) && (idx[2] == N_sub-1) && !(idx[3] == N_sub-1)
                if (box[1,1] <= p[1] <= box[1,2]) & (box[2,1] <= p[2] <= box[2,2]) & (box[3,1] <= p[3] < box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif (idx[1] == N_sub-1) && !(idx[2] == N_sub-1) && (idx[3] == N_sub-1)
                if (box[1,1] <= p[1] <= box[1,2]) & (box[2,1] <= p[2] < box[2,2]) & (box[3,1] <= p[3] <= box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif !(idx[1] == N_sub-1) && (idx[2] == N_sub-1) && (idx[3] == N_sub-1)
                if (box[1,1] <= p[1] < box[1,2]) & (box[2,1] <= p[2] <= box[2,2]) & (box[3,1] <= p[3] <= box[3,2])
                    push!(in_subbox, coord_idx)
                end

            elseif (idx[1] == N_sub-1) && (idx[2] == N_sub-1) && (idx[3] == N_sub-1)
                if (box[1,1] <= p[1] <= box[1,2]) & (box[2,1] <= p[2] <= box[2,2]) & (box[3,1] <= p[3] <= box[3,2])
                    push!(in_subbox, coord_idx)
                end
            
            else ## shouldn't appear if have considered all cases
                println("Warning: unexpected exception in subbox density calulation.")
            end
        end
    end
    return in_subbox
end


## field calculation from subbox estimators -----------------------------------

function density_subbox(coords_arr, ps_dtfe_sb)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    density_arr  = zeros(Float64, size(coords_arr)...)

    box_idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]

    for (i, j, k) in box_idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)

        println("   -> computing " * string(length(in_subbox)) * " elements")
        if length(in_subbox) > 0  ## only load ps-dtfe if necessary
            subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
            @showprogress for idx in in_subbox
                density_arr[idx] = density(coords_arr[idx], subbox_ps_dtfe)
            end
        end
    end
    return density_arr
end


function numberOfStreams_subbox(coords_arr, ps_dtfe_sb)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    nstreams_arr  = zeros(Float64, size(coords_arr)...)

    box_idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]

    for (i, j, k) in box_idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)

        println("   -> computing " * string(length(in_subbox)) * " elements")
        if length(in_subbox) > 0  ## only load ps-dtfe if necessary
            subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
            @showprogress for idx in in_subbox
                nstreams_arr[idx] = numberOfStreams(coords_arr[idx], subbox_ps_dtfe)
            end
        end
    end
    return nstreams_arr
end


function velocity_subbox(coords_arr, ps_dtfe_sb)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    velocity_arr  = zeros(Float64, size(coords_arr)..., 3)

    box_idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]

    for (i, j, k) in box_idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)

        println("   -> computing " * string(length(in_subbox)) * " elements")
        if length(in_subbox) > 0  ## only load ps-dtfe if necessary
            subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
            @showprogress for idx in in_subbox
                #=
                v_ = velocity(coords_arr[idx], subbox_ps_dtfe)
                if size(v_) == (1,3)  # single stream
                    velocity_arr[idx, :] = v_
                else
                    velocity_arr[idx, :] = [NaN, NaN, NaN]
                end
                =#
                velocity_arr[idx, :] = velocity(coords_arr[idx], subbox_ps_dtfe, true)[:]
            end
        end
    end
    return velocity_arr
end

function velocitySum_subbox(coords_arr, ps_dtfe_sb)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    velocity_arr  = zeros(Float64, size(coords_arr)..., 3)

    box_idx_list = [[i, j, k] for i in 0:N_sub-1, j in 0:N_sub-1, k in 0:N_sub-1]

    for (i, j, k) in box_idx_list
        println("subbox i, j, k = " * string([i, j, k]))
        in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)

        println("   -> computing " * string(length(in_subbox)) * " elements")
        if length(in_subbox) > 0  ## only load ps-dtfe if necessary
            subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
            @showprogress for idx in in_subbox
                velocity_arr[idx, :] = velocitySum(coords_arr[idx], subbox_ps_dtfe)
            end
        end
    end
    return velocity_arr
end



## modular subbox calculations: single subbox ---------------------------------

function get_subboxes(ps_dtfe_sub::PS_DTFE_subbox)
    [[i, j, k] for i in 0:ps_dtfe_sub.N_sub-1, j in 0:ps_dtfe_sub.N_sub-1, k in 0:ps_dtfe_sub.N_sub-1][:]
end

function get_coords_chunk(coords, m, m_idx, random=false)
    indices = CartesianIndices(size(coords))
    
    flat_coords  = reshape(coords, :)
    flat_indices = reshape(collect(indices), :)
    println("flat_coords = ", flat_coords[1], " ", flat_coords[end])
    println("flat_indices = ", flat_indices[1], " ", flat_indices[end])

    if random  # permute coordinates for approx. equal computational resources across jobs
        rng = MersenneTwister(1)
        perm = randperm(rng, length(flat_coords))

        flat_coords  = flat_coords[perm]
        flat_indices = flat_indices[perm]
    end

    println("flat_coords = ", flat_coords[1], " ", flat_coords[end])
    println("flat_indices = ", flat_indices[1], " ", flat_indices[end])

    coords_chunks = chunks(flat_coords; n=m)
    index_chunks  = chunks(flat_indices; n=m)

    collect(coords_chunks[m_idx]), collect(index_chunks[m_idx])
end


function density_subbox_single(coords_arr, ps_dtfe_sb, box_idx)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    i, j, k = box_idx
    println("subbox i, j, k = " * string([i, j, k]))

    in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)
    println("   -> computing " * string(length(in_subbox)) * " elements")

    densities = nothing
    if length(in_subbox) > 0  ## only load ps-dtfe if necessary
        subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
        densities = @showprogress [density(coords_arr[idx], subbox_ps_dtfe) for idx in in_subbox]
    end
    
    return densities, in_subbox
end

function numberOfStreams_subbox_single(coords_arr, ps_dtfe_sb, box_idx)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    i, j, k = box_idx
    println("subbox i, j, k = " * string([i, j, k]))

    in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)
    println("   -> computing " * string(length(in_subbox)) * " elements")

    nstreams = nothing
    if length(in_subbox) > 0  ## only load ps-dtfe if necessary
        subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
        nstreams = @showprogress [numberOfStreams(coords_arr[idx], subbox_ps_dtfe) for idx in in_subbox]
    end
    
    return nstreams, in_subbox
end

function velocity_subbox_single(coords_arr, ps_dtfe_sb, box_idx)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    i, j, k = box_idx
    println("subbox i, j, k = " * string([i, j, k]))

    in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)
    println("   -> computing " * string(length(in_subbox)) * " elements")

    vels = nothing
    if length(in_subbox) > 0  ## only load ps-dtfe if necessary
        subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
        vels = @showprogress [velocity(coords_arr[idx], subbox_ps_dtfe, true)[:] for idx in in_subbox]
    end
    
    return vels, in_subbox
end

function velocitySum_subbox_single(coords_arr, ps_dtfe_sb, box_idx)
    N_sub = ps_dtfe_sb.N_sub
    dir   = ps_dtfe_sb.dir

    i, j, k = box_idx
    println("subbox i, j, k = " * string([i, j, k]))

    in_subbox = get_coords_in_subbox(coords_arr, [i, j, k], N_sub, ps_dtfe_sb.L)
    println("   -> computing " * string(length(in_subbox)) * " elements")

    vels = nothing
    if length(in_subbox) > 0  ## only load ps-dtfe if necessary
        subbox_ps_dtfe = deserialize(dir * "/box_" * string(i) * "_" * string(j) * "_" * string(k))
        vels = @showprogress [velocitySum(coords_arr[idx], subbox_ps_dtfe)[:] for idx in in_subbox]
    end
    
    return vels, in_subbox
end