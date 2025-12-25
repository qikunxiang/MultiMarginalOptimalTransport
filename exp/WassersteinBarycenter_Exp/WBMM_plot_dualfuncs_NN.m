% Plot the computed MMOT dual solutions

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);

clim_list = [
    -0.05, 0.15; ...
    -0.005, 0.10; ...
    -0.30, 0.05; ...
    -0.15, 0.10; ...
    -0.15, 0.30];

FILE_PREFIX = 'exp/MMOT/Comparisons/Saved/WassersteinBarycenter/NN/';
FILE_SUFFIX = '_ReLu_5_128.json';
FILE_NAMES_CELL = {
    'L2_50000', ...
    'L2_500000', ...
    'L2_5000000'
};



figure('Position', [200, 100, 800, 1200]);
ha = tight_subplot(marg_num, 3, [0.024, 0.035], [0.024, 0.003], [0.035, 0.0035]);
col_num = ceil(length(ha) / marg_num);

for plot_id = 1:length(FILE_NAMES_CELL)
    json_file_name = [FILE_PREFIX, FILE_NAMES_CELL{plot_id}, FILE_SUFFIX];

    data = readstruct(json_file_name);
    grid_pts = vertcat(data.dual_funcs_eval_points{:});
    grid_x = reshape(grid_pts(:, 1), 201, 201);
    grid_y = reshape(grid_pts(:, 2), 201, 201);
    vals = vertcat(data.dual_funcs_vals{:})';

    % shift the functions by constants such that all but the first function evaluates to 0 at 0
    vals(:, 1) = vals(:, 1) + sum(vals(1, 2:end));
    vals(:, 2:end) = vals(:, 2:end) - vals(1, 2:end);

    for marg_id = 1:marg_num
        
        axes(ha((marg_id - 1) * col_num + plot_id));

        plot_color = pcolor(grid_x, grid_y, reshape(vals(:, marg_id), size(grid_x, 1), size(grid_x, 2)));
        plot_color.EdgeColor = 'interp';
        plot_color.FaceColor = 'interp';

        colormap('jet');
        % clim(clim_list(marg_id, :));

        set(gca, 'XTick', 0:3);
        set(gca, 'YTick', 0:3);
        set(gca, 'XLim', [0.0, 3.0]);
        set(gca, 'YLim', [0.0, 3.0]);

        set(gca, 'XTickLabels', []);
        set(gca, 'YTickLabels', []);
        box on;
        grid on;
    end
end