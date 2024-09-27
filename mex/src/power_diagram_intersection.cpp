#include "power_diagram_intersection.hpp"
#include "kdtree-cpp/kdtree.hpp"

/// @brief Compute the power diagram of a collection of circles with given centers and radii. 
/// Segments in the resulting power diagram are added to a given list and the rays are ignored.
/// Reference: Aurenhammer, F., 1987. Power diagrams: properties, algorithms and applications. SIAM Journal on Computing, 16(1), pp.78-96.
/// @param circles [input] std::vector containing objects of std::pair which contains the centers and squared radii of circles
/// @param segments [output] std::forward_list for appending the segments at the back
void add_power_diagram_segments(const std::vector<std::pair<Point_2, exact_field_t>> & circles, \
    std::forward_list<Segment_2> & segments)
{
    size_t circle_num = circles.size();

    // create the container of the lifted points; these points in 3D are dual to the hyperplanes corresponding to the circles
    std::vector<Point_3> lifted_points;
    lifted_points.reserve(circle_num);

    for (auto circle = circles.begin(); circle != circles.end(); ++circle)
    {
        Point_2 center = circle->first;
        exact_field_t squared_radius = circle->second;

        // the radius need to be non-negative
        CGAL_precondition(squared_radius >= 0);

        // suppose that the circle s has center (s1, s2) and radius r, then the circle corresponds to a hyperplane:
        // h = \Pi(s): x3 = 2 * s1 * x1 + 2 * s2 * x2 - s1^2 - s2^2 + r^2 and \Delta(h) = (s1, s2, s1^2 + s2^2 - r^2);
        // see Section 4.1 of Aurenhammer 1987
        exact_field_t lifted_x = center.x();
        exact_field_t lifted_y = center.y();
        exact_field_t lifted_z = lifted_x * lifted_x + lifted_y * lifted_y - squared_radius;

        // append the lifted point
        lifted_points.push_back(Point_3(lifted_x, lifted_y, lifted_z));
    }

    // create the 3D polyhedron to represent the convex hull of the lifted points
    Polyhedron_3 convhull;

    // compute the convex hull
    CGAL::convex_hull_3(lifted_points.begin(), lifted_points.end(), convhull);


    // create the map for storing the correspondence between facets of the convex hull and points in the power diagram
    std::unordered_map<Facet_3_handle, Point_2> facet_map;

    // traverse the facets of the convex hull
    for (Facet_3_handle facet_handle = convhull.facets_begin(); facet_handle != convhull.facets_end(); ++facet_handle)
    {
        const Facet_3 & facet = *facet_handle;

        // the facet should be a triangle
        CGAL_precondition(facet.is_triangle());

        auto halfedge_handle = facet.halfedge();
        
        // construct a triangle with the three vertices of the facet in order to compute the equation
        Triangle_3 facet_triangle = Triangle_3(halfedge_handle->vertex()->point(), \
            halfedge_handle->prev()->vertex()->point(), \
            halfedge_handle->prev()->prev()->vertex()->point());

        Plane_3 facet_plane = facet_triangle.supporting_plane();

        // skip all facets that are perpendicular to the xy-plane or facing downwards
        if (facet_plane.c() <= 0)
            continue;

        // insert the point in the power diagram that corresponds to the facet
        // suppose that the facet is contained in the hyperplane h: x3 = -a / c * x1 - b / c * x2 - d / c, then 
        // \Deta(h) = (-a / c / 2, -b / c / 2, d / c), and we will discard the third coordinate
        exact_field_t power_diagram_point_x = -facet_plane.a() / facet_plane.c() / 2;
        exact_field_t power_diagram_point_y = -facet_plane.b() / facet_plane.c() / 2;
        facet_map.insert_or_assign(facet_handle, Point_2(power_diagram_point_x, power_diagram_point_y));
        
        // traverse the neighboring facets and store the segments, in exist
        for (size_t halfedge_index = 0; halfedge_index < 3; ++halfedge_index)
        {
            // retrieve the neighboring facet through the opposite halfedge
            Facet_3_handle neighbor_facet_handle = halfedge_handle->opposite()->facet();

            try
            {
                const Point_2 & neighbor_point = facet_map.at(neighbor_facet_handle);
                Segment_2 new_segment = Segment_2(Point_2(power_diagram_point_x, power_diagram_point_y), neighbor_point);

                // add a segment that connects the power diagram point corresponding to the current facet with 
                // the power diagram point corresponding to the neighboring facet
                // only add the segment if it is non-degenerate
                if (!new_segment.is_degenerate())
                    segments.push_front(std::move(new_segment));
            }
            catch (const std::out_of_range e)
            {
                // the neighbor facet has not been traversed yet, or is not facing upwards
            }

            // move to the next halfedge
            halfedge_handle = halfedge_handle->next();
        }
    }
}


/// @brief Compute the polyhedral cell complex formed by the intersection of power diagrams. Rather than returning the resulting polyhedral 
/// cell complex, the centroid of each non-empty cell is returned in a std::vector. 
/// @param segments [input] a list containing all segments in the power diagrams; rays are ignored and thus artificial extra points need to be 
/// added to the construction of power diagrams to truncate the rays to segments
/// @return a std::vector containing the centroid points of the resulting polyhedral cell complex
std::vector<Point_2> compute_power_diagram_intersection(const std::forward_list<Segment_2> & segments)
{
    // build the arrangement from the segments
    Arrangement arrangement;
    insert(arrangement, segments.begin(), segments.end());

    std::vector<Point_2> centroids;
    centroids.reserve(arrangement.number_of_faces());

    for (auto face_handle = arrangement.faces_begin(); face_handle != arrangement.faces_end(); ++face_handle)
    {
        // skip the unbounded face (there should be only 1)
        if (face_handle->is_unbounded())
            continue;

        // start computing the centroid of the face (which is simply the arithmetic mean of the vertices)
        exact_field_t centroid_x = 0;
        exact_field_t centroid_y = 0;
        int face_vertex_counter = 0;

        // traverse around the outer CCB of the face
        auto outer_ccb_handle = face_handle->outer_ccb();
        auto first_vertex_handle = outer_ccb_handle->source();

        while (true)
        {
            centroid_x += outer_ccb_handle->source()->point().x();
            centroid_y += outer_ccb_handle->source()->point().y();
            ++face_vertex_counter;

            // when the next vertex in the traversal equals the first one, the traversal is complete
            if (outer_ccb_handle->target() == first_vertex_handle)
                break;
            
            outer_ccb_handle = outer_ccb_handle->next();
        }

        centroids.push_back(Point_2(centroid_x / face_vertex_counter, centroid_y / face_vertex_counter));
    }

    return centroids;
}

/// @brief Compute the triangular cell complex formed by the intersection of a power diagram and a triangular mesh. 
/// Each non-triangular cell in the resulting intersection is triangulated. The function returns a std::vector containing 
/// the points in the triangulation, a std::vector containing the triangulation indices, a std::vector containing
/// the indices of the power cell each output triangle belongs to, and a std::vector containing the indices in the
/// triangular mesh each output triangle belongs to
/// @param mesh_vertices [input] std::vector containing the vertices in the triangular mesh
/// @param mesh_triangulation [input] std::vector containing the triangulation of the triangular mesh
/// @param circles [input] std::vector containing objects of std::pair which contains the centers and squared radii of circles
/// @param output_vertices [output] std::vector containing the vertices in the output triangulation (there might be irrelevant vertices that do not appear in the triangulation)
/// @param output_triangulation [output] std::vector containing the output triangulation
/// @param power_cell_indices [output] std::vector containing the indices of the power cell each output triangle belongs to
/// @param mesh_indices [output] std::vector containing the indices in the triangular mesh each output triangle belongs to
void compute_mesh_intersect_power_diagram(const std::vector<Point_2> & mesh_vertices, const std::vector<tri_indices_t> & mesh_triangulation, \
    const std::vector<std::pair<Point_2, exact_field_t>> & circles, \
    std::vector<Point_2> & output_vertices, std::vector<tri_indices_t> & output_triangulation, \
    std::vector<size_t> & power_cell_indices, std::vector<size_t> & mesh_indices)
{
    // first construct Triangle_2 objects for each mesh triangle
    std::vector<Triangle_2> mesh_triangles;
    mesh_triangles.reserve(mesh_triangulation.size());

    // create an empty list and add all segments in the power diagrams to it
    std::forward_list<Segment_2> segments;

    for (tri_indices_t tri_indices : mesh_triangulation)
    {
        mesh_triangles.push_back(Triangle_2(mesh_vertices[std::get<0>(tri_indices)], mesh_vertices[std::get<1>(tri_indices)], \
            mesh_vertices[std::get<2>(tri_indices)]));
        
        // append the segments in the triangular mesh
        segments.push_front(Segment_2(mesh_vertices[std::get<0>(tri_indices)], mesh_vertices[std::get<1>(tri_indices)]));
        segments.push_front(Segment_2(mesh_vertices[std::get<1>(tri_indices)], mesh_vertices[std::get<2>(tri_indices)]));
        segments.push_front(Segment_2(mesh_vertices[std::get<2>(tri_indices)], mesh_vertices[std::get<0>(tri_indices)]));
    }

    add_power_diagram_segments(circles, segments);

    // build the arrangement from the segments
    Arrangement arrangement;
    insert(arrangement, segments.begin(), segments.end());

    // store the centroid of each face in order to compute which power cell each face belongs to later
    std::vector<Point_2> centroids;
    centroids.reserve(arrangement.number_of_faces());

    // clear the output data structures before appending the outputs
    output_vertices.clear();
    output_triangulation.clear();
    power_cell_indices.clear();
    mesh_indices.clear();

    // the numbers of vertices and triangles in the output triangulation are at least the same as in the arrangement,
    // therefore, we pre-allocate the space
    output_vertices.reserve(arrangement.number_of_vertices());
    output_triangulation.reserve(arrangement.number_of_faces());

    // this data structure is first used to store the face indices of the triangles; subsequently, once each face is mapped to
    // a power cell via the centroid, we can update these indices such that they will map to the power cell indices
    power_cell_indices.reserve(arrangement.number_of_faces());
    
    // same with the mesh triangle indices
    mesh_indices.reserve(arrangement.number_of_faces());

    // unordered map that maps each vertex to its index in output_vertices
    std::unordered_map<Vertex_2_handle, size_t> output_vertices_map;

    // vector containing sets that store the membership relation between the output vertices and the mesh triangles
    std::vector<std::set<size_t>> output_vertices_membership;
    output_vertices_membership.reserve(arrangement.number_of_vertices());

    size_t output_vertex_counter = 0;

    // traverse all the vertices and register all the vertex handles with their unique indices
    for (Vertex_2_handle vertex_handle = arrangement.vertices_begin(); vertex_handle != arrangement.vertices_end(); ++vertex_handle)
    {
        bool inside_mesh = false;

        // if the vertex does not belong to any of the mesh triangles, it is resulted from the power diagram and can be discarded
        for (size_t triangle_id = 0; triangle_id < mesh_triangles.size(); ++triangle_id)
        {
            if (!mesh_triangles[triangle_id].has_on_unbounded_side(vertex_handle->point()))
            {
                // register the fact that this vertex is contained in the mesh triangle
                if (output_vertices_membership.size() <= output_vertex_counter)
                    output_vertices_membership.push_back(std::set<size_t>());

                output_vertices_membership[output_vertex_counter].insert(triangle_id);

                inside_mesh = true;
            }
        }

        // skip the irrelevant vertices
        if (!inside_mesh)
            continue;

        output_vertices.push_back(vertex_handle->point());
        output_vertices_map.insert_or_assign(vertex_handle, output_vertex_counter);
        ++output_vertex_counter;
    }

    // traverse all the faces
    size_t face_counter = 0;
    for (auto face_handle = arrangement.faces_begin(); face_handle != arrangement.faces_end(); ++face_handle)
    {
        // skip the unbounded face (there should be only 1)
        if (face_handle->is_unbounded())
            continue;

        // for computing the centroid of the face (which is simply the arithmetic mean of the vertices)
        exact_field_t centroid_x = 0;
        exact_field_t centroid_y = 0;
        size_t face_vertex_counter = 0;

        // traverse around the outer CCB of the face
        auto outer_ccb_handle = face_handle->outer_ccb();
        auto first_vertex_handle = outer_ccb_handle->source();

        // keep track of the number of triangles added for each face in case the face is found out to be invalid and all the triangles that are
        // already added need to be removed
        size_t added_triangle_counter = 0;

        try
        {
            size_t first_vertex_index = output_vertices_map.at(first_vertex_handle);

            // start with the set of mesh triangles that the starting vertex is contained in
            std::set<size_t> mesh_triangle_candidates = output_vertices_membership[first_vertex_index];

            while (true)
            {
                size_t edge_source_index = output_vertices_map.at(outer_ccb_handle->source());
                size_t edge_target_index = output_vertices_map.at(outer_ccb_handle->target());

                // since the face is a convex polygon, we adopt the fan triangulation
                if (outer_ccb_handle->source() != first_vertex_handle && outer_ccb_handle->target() != first_vertex_handle)
                {
                    // unless the current edge is adjacent to the beginning vertex, we add a triangle formed by the beginning vertex and the
                    // two ends of the edge

                    // if these vertices are all contained in the mesh, then no exception will be thrown
                    output_triangulation.push_back(tri_indices_t{first_vertex_index, edge_source_index, edge_target_index});

                    // store the face id in power_cell_indices
                    power_cell_indices.push_back(face_counter);

                    ++added_triangle_counter;
                }

                centroid_x += outer_ccb_handle->source()->point().x();
                centroid_y += outer_ccb_handle->source()->point().y();
                ++face_vertex_counter;

                // update the set of possible mesh triangles
                const std::set<size_t> & edge_source_mesh_triangles = output_vertices_membership[edge_source_index];
                auto not_containing_the_vertex = [edge_source_mesh_triangles] (const size_t & x) {return !edge_source_mesh_triangles.contains(x);};
                std::erase_if(mesh_triangle_candidates, not_containing_the_vertex);

                // when the next vertex in the traversal equals the first one, the traversal is complete
                if (outer_ccb_handle->target() == first_vertex_handle)
                    break;
                
                outer_ccb_handle = outer_ccb_handle->next();
            }

            // if there is not a unique mesh triangle that contains all of the vertices in this face, it means that this face is irrelevant; thus, an out_of_range exception is thrown (since it is indeed out of range)
            if (mesh_triangle_candidates.size() != 1)
                throw std::out_of_range("face lies outside the mesh");

            // retrieve the index of this unique mesh triangle
            size_t face_mesh_triangle_index = *(mesh_triangle_candidates.begin());

            // each new output triangle belongs to this mesh triangle
            for (size_t added_triangle_id = 0; added_triangle_id < added_triangle_counter; ++added_triangle_id)
            {
                mesh_indices.push_back(face_mesh_triangle_index);
            }

            centroids.push_back(Point_2(centroid_x / exact_field_t(face_vertex_counter), \
                centroid_y / exact_field_t(face_vertex_counter)));
            ++face_counter;
        }
        catch (const std::out_of_range e)
        {
            // if an exception is thrown, it means the current face is irrelevant and should be discarded;
            // all the triangles that are already added need to be removed
            for (size_t added_triangle_id = 0; added_triangle_id < added_triangle_counter; ++added_triangle_id)
            {
                output_triangulation.pop_back();
                power_cell_indices.pop_back();
            }
        }
    }

    // compute the power cell each centroid belongs to
    std::vector<size_t> centroid_cell_indices = compute_power_diagram_cell_indices(centroids, circles);

    // lastly, traverse the triangles to update the power cell indices
    for (size_t triangle_id = 0; triangle_id < output_triangulation.size(); ++triangle_id)
    {
        power_cell_indices[triangle_id] = centroid_cell_indices[power_cell_indices[triangle_id]];
    }
}

/// @brief Given a collection of 2D points, query which cells in the power diagram they belong to.
/// The use of K-d tree (K-dimensional tree) was inspired by Altschuler, J.M. and Boix-Adserà, E., 2021. Wasserstein barycenters can be computed 
/// in polynomial time in fixed dimension. Journal of Machine Learning Research, 22(44), pp.1-19. See https://github.com/eboix/high_precision_barycenters/tree/master.
/// @param query_points [input] std::vector containing the points to be queried
/// @param circles [input] std::vector containing objects of std::pair which contains the centers and squared radii of circles
/// @return std::vector containing the indices of the power diagram cells that the query points belong to
std::vector<size_t> compute_power_diagram_cell_indices(const std::vector<Point_2> & query_points, \
    const std::vector<std::pair<Point_2, exact_field_t>> & circles)
{
    // first compute the maximum radius of the circles
    double circle_squared_radius_max = -1;

    for (std::pair<Point_2, exact_field_t> circle : circles)
    {
        double current_circle_squared_radius = circle.second.to_double();
        if (current_circle_squared_radius > circle_squared_radius_max)
        {
            circle_squared_radius_max = current_circle_squared_radius;
        }
    }

    // construct a K-d tree for finding the nearest neighbor
    Kdtree::KdNodeVector kdtree_nodes;
    size_t circle_counter = 0;

    for (std::pair<Point_2, exact_field_t> circle : circles)
    {
        // the x- and y-coordinates of the point in the K-d tree are identical to the center of the circle;
        // the z-coordinate of the point in the K-d tree is equal to sqrt(r_max^2 - r^2) where r is the radius of the circle;
        // subsequently, the squared Euclidean distance between (x, y, 0) and (sx, sy, sqrt(radius_max^2 - radius^2)) equals 
        // (x - sx)^2 + (y - sy)^2 + r_max^2 - r^2; since r_max is constant, this is equivalent to using the squared Euclidean distance
        // (x - sx)^2 + (y - sy)^2 - r^2 for determining the nearest neighbor, which corresponds to finding the power diagram cell that 
        // a point belongs to
        double circle_center_x = circle.first.x().to_double();
        double circle_center_y = circle.first.y().to_double();
        double circle_squared_radius = circle.second.to_double();
        kdtree_nodes.push_back(Kdtree::KdNode(std::vector<double>{circle_center_x, circle_center_y, \
            std::sqrt(circle_squared_radius_max - circle_squared_radius)}, NULL, circle_counter));
        ++circle_counter;
    }

    Kdtree::KdTree kdtree(&kdtree_nodes);

    size_t point_num = query_points.size();

    std::vector<size_t> cell_indices;
    cell_indices.reserve(point_num);

    for (Point_2 point : query_points)
    {
        double point_x = point.x().to_double();
        double point_y = point.y().to_double();

        // data structure for storing the query results
        Kdtree::KdNodeVector kdtree_result;
        kdtree.k_nearest_neighbors(std::vector<double>{point_x, point_y, 0.0}, 1, &kdtree_result);

        cell_indices.push_back(kdtree_result[0].index);
    }

    return cell_indices;
}