#ifndef __POWER_DIAGRAM_INTERSECTION_HPP__
#define __POWER_DIAGRAM_INTERSECTION_HPP__

#include <CGAL/Gmpq.h>
#include <CGAL/Simple_cartesian.h>
#include <CGAL/Cartesian_converter.h>
#include <CGAL/Polyhedron_3.h>
#include <CGAL/Point_2.h>
#include <CGAL/Segment_2.h>
#include <CGAL/convex_hull_3.h>
#include <CGAL/Arr_linear_traits_2.h>
#include <CGAL/Arrangement_2.h>
#include <CGAL/assertions.h>
#include <vector>
#include <forward_list>
#include <unordered_map>
#include <tuple>
#include <cassert>
#include <cmath>

using exact_field_t     =   CGAL::Gmpq;
using Kernel            =   CGAL::Simple_cartesian<exact_field_t>;
using Polyhedron_3      =   CGAL::Polyhedron_3<Kernel>;
using Point_3           =   Kernel::Point_3;
using Triangle_3        =   Kernel::Triangle_3;
using Facet_3           =   Polyhedron_3::Facet;
using Facet_3_handle    =   Polyhedron_3::Facet_const_handle;
using Plane_3           =   Polyhedron_3::Plane_3;
using Halfedge_3        =   Polyhedron_3::Halfedge;
using Point_2           =   CGAL::Point_2<Kernel>;
using Segment_2         =   CGAL::Segment_2<Kernel>;
using Triangle_2        =   CGAL::Triangle_2<Kernel>;

using Traits            =   CGAL::Arr_linear_traits_2<Kernel>;
using Arrangement       =   CGAL::Arrangement_2<Traits>;
using Vertex_2_handle   =   Arrangement::Vertex_const_handle;

using tri_indices_t     =   std::tuple<size_t, size_t, size_t>;


/// @brief Compute the power diagram of a collection of circles with given centers and radii. 
/// Segments in the resulting power diagram are added to a given list and the rays are ignored.
/// Reference: Aurenhammer, F., 1987. Power diagrams: properties, algorithms and applications. SIAM Journal on Computing, 16(1), pp.78-96.
/// @param circles std::vector containing objects of std::pair which contains the centers and squared radii of circles
/// @param segments std::forward_list for appending the segments at the back
void add_power_diagram_segments(const std::vector<std::pair<Point_2, exact_field_t>> &circles, std::forward_list<Segment_2> &segments);

/// @brief Compute the polyhedral cell complex formed by the intersection of power diagrams. Rather than returning the resulting polyhedral 
/// cell complex, the centroid of each non-empty cell is returned in a std::vector. 
/// @param segments a list containing all segments in the power diagrams; rays are ignored and thus artificial extra points need to be 
/// added to the construction of power diagrams to truncate the rays to segments
/// @return a std::vector containing the centroid points of the resulting polyhedral cell complex
std::vector<Point_2> compute_power_diagram_intersection(const std::forward_list<Segment_2> &segments);

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
    std::vector<size_t> & power_cell_indices, std::vector<size_t> & mesh_indices);

/// @brief Given a collection of 2D points, query which cells in the power diagram they belong to.
/// The use of K-d tree (K-dimensional tree) was inspired by Altschuler, J.M. and Boix-Adserà, E., 2021. Wasserstein barycenters can be computed 
/// in polynomial time in fixed dimension. Journal of Machine Learning Research, 22(44), pp.1-19. See https://github.com/eboix/high_precision_barycenters/tree/master.
/// @param query_points [input] std::vector containing the points to be queried
/// @param circles [input] std::vector containing objects of std::pair which contains the centers and squared radii of circles
/// @return std::vector containing the indices of the power diagram cells that the query points belong to
std::vector<size_t> compute_power_diagram_cell_indices(const std::vector<Point_2> & query_points, \
    const std::vector<std::pair<Point_2, exact_field_t>> & circles);

#endif // __POWER_DIAGRAM_INTERSECTION_HPP__