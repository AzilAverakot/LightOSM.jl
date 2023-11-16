"""
    astar([::Type{<:AStar},]
          g::AbstractGraph{U},
          weights::AbstractMatrix{T},
          src::W,
          goal::W;
          heuristic::Function=(u, v) ->  0.0,
          cost_adjustment::Function=(u, v, parents) -> 0.0,
          max_distance::T=typemax(T)
          ) where {T <: Real, U <: Integer, W <: Integer}

A* shortest path algorithm. Implemented with a min heap. Using a min heap is 
faster than using a priority queue given the sparse nature of OpenStreetMap 
data, i.e. vertices far outnumber edges.

There are two implementations:
- `AStarVector` is faster for small graphs and/or long paths. This is default. 
    It pre-allocates vectors at the start of the algorithm to store 
    distances, parents and visited nodes. This speeds up graph traversal at the 
    cost of large memory usage.
- `AStarDict` is faster for large graphs and/or short paths.
    It dynamically allocates memory during traversal to store distances, 
    parents and visited nodes. This is faster compared to `AStarVector` when 
    the graph contains a large number of nodes and/or not much traversal is 
    required.

Compared to `jl`, this version improves runtime, memory usage, has a flexible 
heuristic function, and accounts for OpenStreetMap turn restrictions through 
the `cost_adjustment` function.

**Note**: A heuristic that does not accurately estimate the remaining cost to 
`goal` (i.e. overestimating heuristic) will result in a non-optimal path 
(i.e. not the shortest), dijkstra on the other hand guarantees the optimal path 
as the heuristic cost is zero.

# Arguments
- `::Type{<:AStar}`: Implementation to use, either `AStarVector` (default) or 
    `AStarDict`.
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `src::W`: Source vertex.
- `goal::W`: Goal vertex.
- `heuristic::Function=h(u, v) =  0.0`: Heuristic cost function, takes a source 
    and target vertex, default is 0.
- `cost_adjustment:::Function=r(u, v, parents) = 0.0`: Optional cost adjustment 
    function for use cases such as turn restrictions, takes a source and target 
    vertex, defaults to 0.
- `max_distance::T=typemax(T)`: Maximum weight to traverse the graph, returns 
    `nothing` if this is reached.

# Return
- `Union{Nothing,Vector{U}}`: Array veritces represeting shortest path from 
    `src` to `goal`.
"""
function astar(::Type{A},
               g::AbstractGraph{U},
               weights::AbstractMatrix{T},
               src::W,
               goal::W;
               heuristic::Function=(u, v) ->  0.0,
               cost_adjustment::Function=(u, v, parents) -> 0.0,
               max_distance::T=typemax(T)
               ) where {A <: AStar, T <: Real, U <: Integer, W <: Integer}
    # Preallocate
    heap = BinaryHeap{Tuple{T, U, U}}(FastMin) # (f = g + h, current, path length)
    dists = fill(typemax(T), nv(g))
    parents = zeros(U, nv(g))
    visited = zeros(Bool, nv(g))
    len = zero(U)

    # Initialize src
    dists[src] = zero(T)
    push!(heap, (zero(T), src, len))

    while !isempty(heap)
        _, u, len = pop!(heap) # (f = g + h, current, path length)
        visited[u] && continue
        visited[u] = true
        len += one(U)
        u == goal && break # optimal path to goal found
        d = dists[u]
        d > max_distance && return # reached max distance

        for v in outneighbors(g, u)
            visited[v] && continue
            alt = d + weights[u, v] + cost_adjustment(u, v, parents) # turn restriction would imply `Inf` cost adjustment
            
            if alt < dists[v]
                dists[v] = alt
                parents[v] = u
                push!(heap, (alt + heuristic(v, goal), v, len))
            end
        end
    end

    return path_from_parents(parents, goal, len)
end
function astar(::Type{AStarDict},
               g::AbstractGraph{U},
               weights::AbstractMatrix{T},
               src::W,
               goal::W;
               heuristic::Function=(u, v) ->  0.0,
               cost_adjustment::Function=(u, v, parents) -> 0.0,
               max_distance::T=typemax(T)
               ) where {T <: Real, U <: Integer, W <: Integer}
    # Preallocate
    heap = BinaryHeap{Tuple{T, U, U}}(FastMin) # (f = g + h, current, path length)
    dists = Dict{U, T}()
    parents = Dict{U, U}()
    visited = Set{U}()
    len = zero(U)

    # Initialize src
    dists[src] = zero(T)
    push!(heap, (zero(T), src, len))

    while !isempty(heap)
        _, u, len = pop!(heap) # (f = g + h, current, path length)
        u in visited && continue
        push!(visited, u)
        len += one(U)
        u == goal && break # optimal path to goal found
        d = get(dists, u, typemax(T))
        d > max_distance && return # reached max distance

        for v in outneighbors(g, u)
            v in visited && continue
            alt = d + weights[u, v] + cost_adjustment(u, v, parents) # turn restriction would imply `Inf` cost adjustment
            
            if alt < get(dists, v, typemax(T))
            dists[v] = alt
                parents[v] = u
                push!(heap, (alt + heuristic(v, goal), v, len))
            end
        end
    end

    return path_from_parents(parents, goal, len)
end
function astar(g::AbstractGraph{U},
               weights::AbstractMatrix{T},
               src::W,
               goal::W;
               kwargs...
               ) where {T <: Real, U <: Integer, W <: Integer}
    return astar(AStarVector, g, weights, src, goal; kwargs...)
end

"""
    dijkstra([::Type{<:Dijkstra},]
            g::AbstractGraph{U},
            weights::AbstractMatrix{T},
            src::W,
            goal::W;
            cost_adjustment::Function=(u, v, parents) -> 0.0,
            max_distance::T=typemax(T)
            ) where {T <: Real, U <: Integer, W <: Integer}

Dijkstra's shortest path algorithm with an early exit condition, is the same as 
astar with heuristic cost as 0.

There are two implementations:
- `DijkstraVector` is faster for small graphs and/or long paths. This is default. 
    It pre-allocates vectors at the start of the algorithm to store 
    distances, parents and visited nodes. This speeds up graph traversal at the 
    cost of large memory usage.
- `DijkstraDict` is faster for large graphs and/or short paths.
    It dynamically allocates memory during traversal to store distances, 
    parents and visited nodes. This is faster compared to `AStarVector` when 
    the graph contains a large number of nodes and/or not much traversal is 
    required.

# Arguments
- `::Type{<:Dijkstra}`: Implementation to use, either `DijkstraVector` 
    (default) or `DijkstraDict`.
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `src::W`: Source vertex.
- `goal::W`: Goal vertex.
- `cost_adjustment:::Function=r(u, v, parents) = 0.0`: Optional cost adjustment 
    function for use cases such as turn restrictions, takes a source and target 
    vertex, defaults to 0.
- `max_distance::T=typemax(T)`: Maximum weight to traverse the graph, returns 
    `nothing` if this is reached.

# Return
- `Union{Nothing,Vector{U}}`: Array veritces represeting shortest path between `src` to `goal`.
"""
function dijkstra(::Type{A},
                  g::AbstractGraph{U},
                  weights::AbstractMatrix{T},
                  src::W,
                  goal::W;
                  kwargs...
                  ) where {A <: Dijkstra, T <: Real, U <: Integer, W <: Integer}
    return astar(AStarVector, g, weights, src, goal; kwargs...)
end
function dijkstra(::Type{DijkstraDict},
                  g::AbstractGraph{U},
                  weights::AbstractMatrix{T},
                  src::W,
                  goal::W;
                  kwargs...
                  ) where {T <: Real, U <: Integer, W <: Integer}
    return astar(AStarDict, g, weights, src, goal; kwargs...)
end
function dijkstra(g::AbstractGraph{U},
                  weights::AbstractMatrix{T},
                  src::W,
                  goal::W;
                  kwargs...
                  ) where {T <: Real, U <: Integer, W <: Integer}
    return dijkstra(DijkstraVector, g, weights, src, goal; kwargs...)
end

"""
    dijkstra(g::AbstractGraph{U},
             weights::AbstractMatrix{T},
             src::W;
             cost_adjustment::Function=(u, v, parents) -> 0.0
             ) where {T <: Real, U <: Integer, W <: Integer}

Dijkstra's shortest path algorithm, implemented with a min heap. Using a min heap is faster than using 
a priority queue given the sparse nature of OpenStreetMap data, i.e. vertices far outnumber edges.

This dispatch returns full set of `parents` or the `dijkstra state` given a source vertex, i.e. without
and early exit condition of `goal`.

# Arguments
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `src::W`: Source vertex.
- `cost_adjustment:::Function=r(u, v, parents) = 0.0`: Optional cost adjustment function for use cases such as turn restrictions, takes a source and target vertex, defaults to 0.

# Return
- `Vector{U}`: Array parent veritces from which the shortest path can be extracted.
"""
function dijkstra(g::AbstractGraph{U},
                  weights::AbstractMatrix{T},
                  src::W;
                  cost_adjustment::Function=(u, v, parents) -> 0.0
                  ) where {T <: Real, U <: Integer, W <: Integer}
    # Preallocate
    heap = BinaryHeap{Tuple{T, U}}(FastMin) # (weight, current)
    dists = fill(typemax(T), nv(g))
    parents = zeros(U, nv(g))
    visited = zeros(Bool, nv(g))

    # Initialize src
    push!(heap, (zero(T), src))
    dists[src] = zero(T)

    while !isempty(heap)
        _, u = pop!(heap) # (weight, current)
        visited[u] && continue
        visited[u] = true
        d = dists[u]

        for v in outneighbors(g, u)
            visited[v] && continue
            alt = d + weights[u, v] + cost_adjustment(u, v, parents) # turn restriction would imply `Inf` cost adjustment
            
            if alt < dists[v]
            dists[v] = alt
                parents[v] = u
                push!(heap, (alt, v))
            end
        end
    end

    return parents
end

"""
    path_from_parents(parents::P, goal::V) where {P <: Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}} where {U <: Integer, V <: Integer}

Extracts shortest path given dijkstra parents of a given source.

# Arguments
- `parents::Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}`: Mapping of 
    dijkstra parent states.
- `goal::V`: Goal vertex.

# Return
- `Union{Nothing,Vector{U}}`: Array veritces represeting shortest path to `goal`.
"""
function path_from_parents(parents::P, goal::V) where {P <: Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}} where {U <: Integer, V <: Integer}
    parents[goal] == 0 && return
    
    pointer = goal
    path = U[]
    
    while pointer != 0 # parent of origin is always 0
        push!(path, pointer)
        pointer = parents[pointer]
    end

    return reverse(path)
end

"""
    path_from_parents(parents::P, goal::V, path_length::N) where {P <: Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}} where {U <: Integer, V <: Integer, N <: Integer}

Extracts shortest path given dijkstra parents of a given source, providing `path_length` allows
preallocation of the array and avoids the need to reverse the path.

# Arguments
- `parents::Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}`: Mapping of dijkstra parent states.
- `goal::V`: Goal vertex.
- `path_kength::N`: Known length of the return path, allows preallocation of final path array.

# Return
- `Union{Nothing,Vector{U}}`: Array veritces represeting shortest path to `goal`.
"""
function path_from_parents(parents::P, goal::V, path_length::N) where {P <: Union{<:AbstractVector{<:U}, <:AbstractDict{<:U, <:U}}} where {U <: Integer, V <: Integer, N <: Integer}
    get(parents, goal, zero(U)) == 0 && return
    
    pointer = goal
    path = Vector{U}(undef, path_length)

    for i in one(U):(path_length - 1)
        path[path_length - i + one(U)] = pointer
        pointer = parents[pointer]
    end
    path[1] = pointer

    return path
end

"""
get_nearest_intersection_details(
        g::OSMGraph{U},
        weights::AbstractMatrix{T},
        source_point::Vector{Float64},
        radius_in_meteres::R
        ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}

Top level function that finds nearest intersection, direction from the intersection to the given point and the path length.

# Arguments
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `source_point::Vector{Float64`: Point on the way to find nearest node to.
- `radius_in_meteres::R`: Maximum straight line haversine distance from the given point up to intersection.

# Return
- `intersection{Integer}`: Index of the start node on the graph.
- `direction{String}`: Index of the parent node through which the intersection was reached. Used to calculate the direction.
- `distance{Float64}`: Array parent veritces from which the shortest path can be extracted.
"""
function get_nearest_intersection_details(
    g::OSMGraph{U},
    weights::AbstractMatrix{T},
    source_point::Vector{Float64},
    radius_in_meteres::R=1000
    ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}
# Get nearest node to the given point
n_start = nearest_node(g, source_point)[1]
n_start_index = g.node_to_index[n_start]
return get_nearest_intersection_details(g, weights, n_start_index,radius_in_meteres)
end
"""
get_nearest_intersection_details(
        g::OSMGraph{U},
        weights::AbstractMatrix{T},
        source_point::Vector{Float64},
        radius_in_meteres::R
        ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}

Top level function that finds nearest intersection, direction from the intersection to the given point and the path length.
NOTE: Takes nod index as input, not ID! 

# Arguments
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `source_point::Vector{Float64`: Point on the way to find nearest node to.
- `radius_in_meteres::R`: Maximum straight line haversine distance from the given point up to intersection.

# Return
- `intersection{Integer}`: Index of the start node on the graph.
- `direction{String}`: Index of the parent node through which the intersection was reached. Used to calculate the direction.
- `distance{Float64}`: Array parent veritces from which the shortest path can be extracted.
"""
function get_nearest_intersection_details(
            g::OSMGraph{U},
            weights::AbstractMatrix{T},
            n_start_index::W,
            radius_in_meteres::R=1000
            ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}
  intersection, parent, distance = dijkstra_nearest_intersection(g, weights, n_start_index, radius_in_meteres)
  direction = calculate_direction(g.node_coordinates[intersection], g.node_coordinates[parent])
  return intersection, direction, distance
end


"""
  dijkstra_nearest_intersection(
          g::OSMGraph{U},
          weights::AbstractMatrix{T},
          source_node_index::W,
          radius_in_meteres::R
          ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}

Dijkstra's shortest path algorithm implemented to loop through all the possible paths from the given node 
withing the given radius to find nearest intersection.
Disregards the directionality of the paths to find the closest intersection by way length. 
Returns index of the intersection node, parent node to be used to find the direction and path length in meters. 

# Arguments
- `g::AbstractGraph{U}`: Graphs abstract graph object.
- `weights::AbstractMatrix{T}`: Edge weights matrix.
- `source_node_index::W`: Source node index.
- `radius_in_meteres::R`: Maximum straight line haversine distance from the given point up to intersection.

# Return
- `node_index{W}`: Index of the start node on the graph.
- `previous_node[node_index]{W}`: Index of the parent node through which the intersection was reached. Used to calculate the direction.
- `dists[node_index]{W}`: Array parent veritces from which the shortest path can be extracted.
"""
function dijkstra_nearest_intersection(
        g::OSMGraph{U},
        weights::AbstractMatrix{T},
        source_node_index::W,
        radius_in_meteres::R=1000
        ) where {T <: Real, U <: Integer, W <: Integer, R <: Integer}
# Preallocate
heap = BinaryHeap{Tuple{T, U}}(FastMin) # (weight, current)
num_verts = nv(g.graph)
dists = fill(typemax(T), num_verts) # Prefilling with infinities to later replace with distances, might not need it
previous_node = zeros(U, num_verts)
visited = zeros(Bool, num_verts) # Can use set
# Initialize src
push!(heap, (zero(T), source_node_index))
dists[source_node_index] = zero(T)
previous_node[source_node_index] = source_node_index
while !isempty(heap)
  weight, node_index = pop!(heap)
  node = g.index_to_node[node_index]
  # Check if this node is an intersection and return this node, previous node (to find direction), distance between source and this node
  haskey(g.nodes[node].tags, "intersections") && return node_index, previous_node[node_index], dists[node_index]
  visited[node_index] && continue
  visited[node_index] = true
  
  # Note: can't check inside this loop if it's an intersection since it doesn't guarantee having the shortest path length
  for neighbor_index in all_neighbors(g.graph, node_index)
      # Check if it has been visited
      visited[neighbor_index] && continue
      # Check if it's inside of the radius
      haversine(g.node_coordinates[neighbor_index], g.node_coordinates[node_index]) > radius_in_meteres && continue
      new_dist_to_neighbor = weight + max(weights[node_index, neighbor_index], weights[neighbor_index, node_index]) # Calculate new neighbor's weight as the weight of the current node and the neighbor's node
      # Check if existing distance to the neighbor is shorter than the previous one
      if new_dist_to_neighbor < dists[neighbor_index]
          dists[neighbor_index] = new_dist_to_neighbor
          previous_node[neighbor_index] = node_index # Save previous node index as a parent_node
          push!(heap, (new_dist_to_neighbor, neighbor_index))
      end
  end
end
# If no intersection is found
return nothing
end