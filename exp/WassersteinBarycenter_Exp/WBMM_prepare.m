% Example illustrating the Wasserstein barycenter problem when all measures
% belong to the same location-scatter family

CONFIG = WBMM_config();

% the input measures have the same support and continuous piece-wise affine
% density functions
meas_x_min = 0;
meas_x_max = 3;
meas_y_min = 0;
meas_y_max = 3;
meas_grid_num_x = 4;
meas_grid_num_y = 4;

meas_x_pts = linspace(meas_x_min, meas_x_max, meas_grid_num_x)';
meas_y_pts = linspace(meas_y_min, meas_y_max, meas_grid_num_y)';
meas_mesh_size = [meas_x_pts(2) - meas_x_pts(1), ...
    meas_y_pts(2) - meas_y_pts(1)];

[meas_grid_x, meas_grid_y] = meshgrid(meas_x_pts, meas_y_pts);

meas_vertices =[meas_grid_x(:), meas_grid_y(:)];
meas_vertices_num = size(meas_vertices, 1);
meas_triangles = zeros((meas_grid_num_x - 1) * (meas_grid_num_y - 1) ...
    * 2, 3);
meas_triangles_num = size(meas_triangles, 1);

tri_counter = 1;
for col_id = 1:meas_grid_num_y - 1
    for row_id = 1:meas_grid_num_x - 1
        meas_triangles(tri_counter, 1) = (row_id - 1) * meas_grid_num_y ...
            + col_id;
        meas_triangles(tri_counter, 2) = meas_triangles(tri_counter, 1) ...
            + 1;
        meas_triangles(tri_counter, 3) = meas_triangles(tri_counter, 1) ...
            + meas_grid_num_y;
        
        meas_triangles(tri_counter + 1, 1) = ...
            meas_triangles(tri_counter, 3) + 1;
        meas_triangles(tri_counter + 1, 2) = ...
            meas_triangles(tri_counter, 3);
        meas_triangles(tri_counter + 1, 3) = ...
            meas_triangles(tri_counter, 2);

        tri_counter = tri_counter + 2;
    end
end

marg_num = 5;

% the weights are set to be all equal
marg_weights = ones(marg_num, 1) / marg_num;

mesh_sizes = [1; 1/2; 1/4; 1/5; 1/8; 1/10; 1/16; 1/20; 1/25];
test_num = length(mesh_sizes);

rs = RandStream("combRecursive", "Seed", 5000);

marg_vertices_cell = cell(marg_num, 1);
marg_triangles_cell = cell(marg_num, 1);
marg_density_cell = cell(marg_num, 1);

for marg_id = 1:marg_num
    marg_vertices_cell{marg_id} = meas_vertices;
    marg_triangles_cell{marg_id} = meas_triangles;
    marg_density_cell{marg_id} = rand(rs, meas_vertices_num, 1) * 10;
end

marg_testfuncs_cell = cell(test_num, 1);

for test_id = 1:test_num
    marg_tf_vert_cell = cell(meas_triangles_num, 1);
    marg_tf_tri = cell(meas_triangles_num, 1);

    pts_list = (0:mesh_sizes(test_id):1)';
    pts_num = length(pts_list);
    [Gx, Gy] = meshgrid(pts_list, pts_list);
    grid1 = [Gx(:), Gy(:)];
    inside1 = sum(grid1, 2) < 1 + 1e-5;
    grid1 = grid1(inside1, :);
    tri1 = delaunay(grid1);
    tri1 = cleanup_triangles(grid1, tri1, [], false);
    
    tri_counter = 1;
    for col_id = 1:meas_grid_num_y - 1
        for row_id = 1:meas_grid_num_x - 1
            marg_tf_vert_cell{tri_counter} = grid1 .* meas_mesh_size ...
                + meas_vertices((row_id - 1) * meas_grid_num_y ...
                + col_id, :);
            marg_tf_tri{tri_counter} = tri1;

            marg_tf_vert_cell{tri_counter + 1} = ...
                (1 - grid1) .* meas_mesh_size ...
                + meas_vertices((row_id - 1) * meas_grid_num_y ...
                + col_id, :);
            marg_tf_tri{tri_counter + 1} = tri1;

            tri_counter = tri_counter + 2;
        end
    end

    marg_tf_vert_agg = [meas_vertices; vertcat(marg_tf_vert_cell{:})];
    [~, uind, umap] = unique(round(marg_tf_vert_agg, 7), 'rows', 'stable');
    marg_tf_vert_unique = marg_tf_vert_agg(uind, :);

    counter = meas_vertices_num;

    for tri_id = 1:meas_triangles_num
        vert_num = size(marg_tf_vert_cell{tri_id}, 1);
        umap_tri = umap(counter + (1:vert_num), :);

        marg_tf_tri{tri_id} = ...
            umap_tri(marg_tf_tri{tri_id});

        if size(marg_tf_tri{tri_id}, 2) == 1
            marg_tf_tri{tri_id} = marg_tf_tri{tri_id}';
        end

        counter = counter + vert_num;
    end

    marg_tf = {marg_tf_vert_unique, marg_tf_tri, false};

    marg_testfuncs_cell{test_id} = cell(marg_num, 1);
    for marg_id = 1:marg_num
        marg_testfuncs_cell{test_id}{marg_id} = marg_tf;
    end
end

% grid used to plot the density of input measures
meas_plot_x_num = 1001;
meas_plot_y_num = 1001;

plot_x_pts = linspace(meas_x_min, meas_x_max, meas_plot_x_num)';
plot_y_pts = linspace(meas_y_min, meas_y_max, meas_plot_y_num)';

[meas_plot_grid_x, meas_plot_grid_y] = meshgrid(plot_x_pts, plot_y_pts);

meas_plot_grid = [meas_plot_grid_x(:), meas_plot_grid_y(:)];


% grid used to plot the histogram of the Wasserstein barycenter
WB_hist_x_num = 200;
WB_hist_y_num = 200;

WB_hist_edge_x = linspace(meas_x_min, meas_x_max, ...
    WB_hist_x_num + 1)';
WB_hist_edge_y = linspace(meas_y_min, meas_y_max, ...
    WB_hist_y_num + 1)';
WB_plot_hist_x = (WB_hist_edge_x(1:end - 1) ...
    + WB_hist_edge_x(2:end)) / 2;
WB_plot_hist_y = (WB_hist_edge_y(1:end - 1) ...
    + WB_hist_edge_y(2:end)) / 2;
[WB_plot_hist_grid_x, WB_plot_hist_grid_y] = ...
    meshgrid(WB_plot_hist_x, WB_plot_hist_y);
WB_plot_hist_grid = [WB_plot_hist_grid_x(:), ...
    WB_plot_hist_grid_y(:)];

save(CONFIG.SAVEPATH_INPUTS, ...
    'test_num', ...
    'marg_num', ...
    'marg_weights', ...
    'marg_vertices_cell', ...
    'marg_triangles_cell', ...
    'marg_density_cell', ...
    'marg_testfuncs_cell', ...
    'meas_plot_x_num', ...
    'meas_plot_y_num', ...
    'meas_plot_grid', ...
    'meas_plot_grid_x', ...
    'meas_plot_grid_y', ...
    'WB_hist_x_num', ...
    'WB_hist_y_num', ...
    'WB_hist_edge_x', ...
    'WB_hist_edge_y', ...
    'WB_plot_hist_grid_x', ...
    'WB_plot_hist_grid_y', ...
    'WB_plot_hist_grid', ...
    '-v7.3');


