% Plot the computed MMOT dual solutions

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_DUALFUNCS);


COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES = [3, 6, 9];

COMPARISONS_OURALGO_TITLES_CELL = cell(length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES), 1);
for test_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    COMPARISONS_OURALGO_TITLES_CELL{test_id} = sprintf('\\textbf{our algo.} ($m_0=%d$)', ...
        size(marg_testfuncs_cell{COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(test_id)}{1}{1}, 1) - 1);
end

COMPARISONS_OURALGO_COLUMN_NUM = 3;

COMPARISONS_RKHS_FILEPREFIX = 'RKHS/';
COMPARISONS_RKHS_FILESUFFIX = '_Laplace_8.0_exp.json';

COMPARISONS_RKHS_FILENAMES_CELL = {
    'L2_25000', ... 
    'L2_50000', ...
    'L2_100000'
};
COMPARISONS_RKHS_TITLES_CELL = {
    'RKHS ($\gamma=2.5\times 10^{4}$)'; ...
    'RKHS ($\gamma=5.0\times 10^{4}$)'; ...
    'RKHS ($\gamma=10^{5}$)'
};
COMPARISONS_RKHS_COLUMN_NUM = 3;

COMPARISONS_NN_FILEPREFIX = 'NN/';
COMPARISONS_NN_FILESUFFIX = '_ReLu_5_128.json';

COMPARISONS_NN_FILENAMES_CELL = {
    'L2_50000', ... 
    'L2_500000', ...
    'L2_5000000'
};
COMPARISONS_NN_TITLES_CELL = {
    'NN ($\gamma=5.0\times 10^{4}$)'; ...
    'NN ($\gamma=5.0\times 10^{5}$)'; ...
    'NN ($\gamma=5.0\times 10^{6}$)'
};
COMPARISONS_NN_COLUMN_NUM = 3;

COMPARISONS_TOTAL_COLUMN_NUM = COMPARISONS_OURALGO_COLUMN_NUM + COMPARISONS_RKHS_COLUMN_NUM + COMPARISONS_NN_COLUMN_NUM;
COMPARISONS_TITLES_CELL = vertcat(COMPARISONS_OURALGO_TITLES_CELL, COMPARISONS_RKHS_TITLES_CELL, COMPARISONS_NN_TITLES_CELL);

clim_ouralgo = [
    -0.05, 0.15; ...
     0.00, 0.09; ...
    -0.28, 0.00; ...
    -0.12, 0.08; ...
    -0.15, 0.26];

clim_RKHS = [
     0.00, 0.23; ...
    -0.10, 0.14; ...
    -0.10, 0.12; ...
    -0.15, 0.01; ...
    -0.15, 0.13];

clim_NN = [
     0.40, 1.00; ...
    -0.36, 0.10; ...
    -0.85, 0.00; ...
    -0.58, 0.00; ...
    -0.20, 0.50];

clim_cell = vertcat(repmat({clim_ouralgo}, COMPARISONS_OURALGO_COLUMN_NUM, 1), ...
    repmat({clim_RKHS}, COMPARISONS_RKHS_COLUMN_NUM, 1), ...
    repmat({clim_NN}, COMPARISONS_NN_COLUMN_NUM, 1));


funcs_cell = cell(COMPARISONS_TOTAL_COLUMN_NUM, 1);

col_counter = 0;

% dual functions computed by our algorithm
for test_id = COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES
    func = struct;
    func.input_grid_x = input_grid_x;
    func.input_grid_y = input_grid_y;
    func.values = MMOT_dualfuncs_cell{test_id};
    col_counter = col_counter + 1;
    funcs_cell{col_counter} = func;
end

% dual functions computed by the RKHS-based algorithm
for test_id = 1:length(COMPARISONS_RKHS_FILENAMES_CELL)
    json_filename = [CONFIG.COMPARISONPATH, COMPARISONS_RKHS_FILEPREFIX, ...
        COMPARISONS_RKHS_FILENAMES_CELL{test_id}, COMPARISONS_RKHS_FILESUFFIX];

    json_data = readstruct(json_filename);
    func = struct;
    grid_pts = vertcat(json_data.dual_funcs_eval_points{:});
    func.input_grid_x = reshape(grid_pts(:, 1), 201, 201);
    func.input_grid_y = reshape(grid_pts(:, 2), 201, 201);
    func.values = vertcat(json_data.dual_funcs_vals{:})';
    func.inputs = json_data.dual_funcs_eval_points';
    func.values = vertcat(json_data.dual_funcs_vals{:})';
    col_counter = col_counter + 1;
    funcs_cell{col_counter} = func;
end

% dual functions computed by the NN-based algorithm
for test_id = 1:length(COMPARISONS_NN_FILENAMES_CELL)
    json_filename = [CONFIG.COMPARISONPATH, COMPARISONS_NN_FILEPREFIX, ...
        COMPARISONS_NN_FILENAMES_CELL{test_id}, COMPARISONS_NN_FILESUFFIX];

    json_data = readstruct(json_filename);
    func = struct;
    grid_pts = vertcat(json_data.dual_funcs_eval_points{:});
    func.input_grid_x = reshape(grid_pts(:, 1), 201, 201);
    func.input_grid_y = reshape(grid_pts(:, 2), 201, 201);
    func.values = vertcat(json_data.dual_funcs_vals{:})';
    func.inputs = json_data.dual_funcs_eval_points';
    func.values = vertcat(json_data.dual_funcs_vals{:})';
    col_counter = col_counter + 1;
    funcs_cell{col_counter} = func;
end

for col_id = 1:COMPARISONS_TOTAL_COLUMN_NUM
    vals = funcs_cell{col_id}.values;

    % shift the functions by constants such that all but the first function evaluates to 0 at the first input point
    vals(:, 1) = vals(:, 1) + sum(vals(1, 2:end));
    vals(:, 2:end) = vals(:, 2:end) - vals(1, 2:end);

    funcs_cell{col_id}.values = vals;
end


figure('Position', [200, 100, 1200, 640]);
ha = tight_subplot(marg_num, COMPARISONS_TOTAL_COLUMN_NUM, [0.006, 0.004], [0.006, 0.023], [0.021, 0.003]);

for col_id = 1:COMPARISONS_TOTAL_COLUMN_NUM
    
    func = funcs_cell{col_id};
    input_grid_x = func.input_grid_x;
    input_grid_y = func.input_grid_y;
    vals = func.values;

    % shift the functions by constants such that all but the first function evaluates to 0 at the first input point
    vals(:, 1) = vals(:, 1) + sum(vals(1, 2:end));
    vals(:, 2:end) = vals(:, 2:end) - vals(1, 2:end);

    for marg_id = 1:marg_num
        
        axes(ha((marg_id - 1) * COMPARISONS_TOTAL_COLUMN_NUM + col_id));

        plot_color = pcolor(input_grid_x, input_grid_y, reshape(vals(:, marg_id), size(input_grid_x, 1), size(input_grid_x, 2)));
        plot_color.EdgeColor = 'interp';
        plot_color.FaceColor = 'interp';

        colormap('jet');
        clim(clim_cell{col_id}(marg_id, :));

        set(gca, 'XTick', 0:3);
        set(gca, 'YTick', 0:3);
        set(gca, 'XLim', [0.0, 3.0]);
        set(gca, 'YLim', [0.0, 3.0]);

        set(gca, 'XTickLabels', []);
        set(gca, 'YTickLabels', []);
        box on;
        grid on;

        if marg_id == 1
            title(COMPARISONS_TITLES_CELL{col_id}, 'Interpreter', 'latex', 'FontSize', 10)
        end
    end
end

annotation('textbox', [0.002, 0.90, 0, 0], 'string', '$\tilde{h}_1$', 'FontSize', 15, 'Interpreter', 'latex');
annotation('textbox', [0.002, 0.71, 0, 0], 'string', '$\tilde{h}_2$', 'FontSize', 15, 'Interpreter', 'latex');
annotation('textbox', [0.002, 0.52, 0, 0], 'string', '$\tilde{h}_3$', 'FontSize', 15, 'Interpreter', 'latex');
annotation('textbox', [0.002, 0.33, 0, 0], 'string', '$\tilde{h}_4$', 'FontSize', 15, 'Interpreter', 'latex');
annotation('textbox', [0.002, 0.14, 0, 0], 'string', '$\tilde{h}_5$', 'FontSize', 15, 'Interpreter', 'latex');