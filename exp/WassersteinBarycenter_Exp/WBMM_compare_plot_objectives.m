% Plot the computed MMOT dual functions

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_UB);
load(CONFIG.SAVEPATH_W2OTUB);

COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES = 9;
COMPARISONS_OURALGO_TEST_NUM = length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES);

COMPARISONS_OURALGO_LEGEND = cell(COMPARISONS_OURALGO_TEST_NUM * 2, 1);
for test_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    COMPARISONS_OURALGO_LEGEND{COMPARISONS_OURALGO_TEST_NUM * (test_id - 1) + 1} = ...
        '$\alpha^{\mathsf{UB}}$';
    COMPARISONS_OURALGO_LEGEND{COMPARISONS_OURALGO_TEST_NUM * (test_id - 1) + 2} = ...
        '$\alpha^{\mathsf{LB}}$';
end
COMPARISONS_LEGEND = vertcat({'RKHS'; 'NN-dual'}, COMPARISONS_OURALGO_LEGEND);

COMPARISONS_SAMPLE_NUM = 20;

COMPARISONS_RKHS_FILEPREFIX = 'RKHS/';
COMPARISONS_RKHS_FILESUFFIX = '_Laplace_8.0_exp_objectives.txt';

COMPARISONS_RKHS_FILENAMES_CELL = {
    'L2_25000', ... 
    'L2_50000', ...
    'L2_100000'
};
COMPARISONS_RKHS_LABELS_CELL = {
    '$\gamma=2.5\times 10^{4}$'; ...
    '$\gamma=5.0\times 10^{4}$'; ...
    '$\gamma=10^{5}$'
};
COMPARISONS_RKHS_COLUMN_NUM = 3;

COMPARISONS_NN_FILEPREFIX = 'NN/';
COMPARISONS_NN_FILESUFFIX_DUAL = '_ReLu_5_128_objectives.txt';

COMPARISONS_NN_FILENAMES_CELL = {
    'L2_50000', ... 
    'L2_500000', ...
    'L2_5000000'
};
COMPARISONS_NN_LABELS_CELL = {
    '$\gamma=5.0\times 10^{4}$'; ...
    '$\gamma=5.0\times 10^{5}$'; ...
    '$\gamma=5.0\times 10^{6}$'
};
COMPARISONS_NN_DUAL_COLUMN_NUM = 3;

COMPARISONS_TOTAL_COLUMN_NUM = COMPARISONS_RKHS_COLUMN_NUM + COMPARISONS_NN_DUAL_COLUMN_NUM;
COMPARISONS_GROUPING = [ones(COMPARISONS_RKHS_COLUMN_NUM, 1); ...
    2 * ones(COMPARISONS_NN_DUAL_COLUMN_NUM, 1)];
COMPARISONS_LABELS_CELL = vertcat(COMPARISONS_RKHS_LABELS_CELL, COMPARISONS_NN_LABELS_CELL);
COMPARISONS_XLIM = [0.2, COMPARISONS_TOTAL_COLUMN_NUM + 0.8];
COMPARISONS_YLIM = [-0.920, 0.220];


bounds = zeros(length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES), 2);

% lower and upper bounds computed by our algorithm
for tf_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    test_id = COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(tf_id);
    bounds(tf_id, 1) = MMOT_LB_list(test_id);
    bounds(tf_id, 2) = MMOT_UB_mean_list(test_id);
end

obj_mat = zeros(COMPARISONS_SAMPLE_NUM, COMPARISONS_TOTAL_COLUMN_NUM);
col_counter = 0;

% objectives computed by the RKHS-based algorithm
for test_id = 1:length(COMPARISONS_RKHS_FILENAMES_CELL)
    txt_filename = [CONFIG.COMPARISONPATH, COMPARISONS_RKHS_FILEPREFIX, ...
        COMPARISONS_RKHS_FILENAMES_CELL{test_id}, COMPARISONS_RKHS_FILESUFFIX];

    txt_file = fopen(txt_filename, 'r');
    samps = fscanf(txt_file, '%f');
    fclose(txt_file);

    col_counter = col_counter + 1;
    obj_mat(:, col_counter) = samps;
end

% entropic dual objectives computed by the NN-based algorithm
for test_id = 1:length(COMPARISONS_NN_FILENAMES_CELL)
    txt_filename = [CONFIG.COMPARISONPATH, COMPARISONS_NN_FILEPREFIX, ...
        COMPARISONS_NN_FILENAMES_CELL{test_id}, COMPARISONS_NN_FILESUFFIX_DUAL];

    txt_file = fopen(txt_filename, 'r');
    samps = fscanf(txt_file, '%f');
    fclose(txt_file);

    col_counter = col_counter + 1;
    obj_mat(:, col_counter) = samps;
end


figure('Position', [0, 0, 480, 300]);
ha = tight_subplot(1, 1, [0, 0], [0.055, 0.005], [0.054, 0.008]);

axes(ha(1));

hold on;
box on;

BC = boxchart(repelem((1:COMPARISONS_TOTAL_COLUMN_NUM)', COMPARISONS_SAMPLE_NUM, 1), obj_mat(:), ...
    'GroupByColor', repelem(COMPARISONS_GROUPING, COMPARISONS_SAMPLE_NUM, 1));

for tf_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    bounds_tf = bounds(tf_id, :);

    if tf_id == 1
        bound_color = 'blue';
    else
        bound_color = 'red';
    end

    plot(COMPARISONS_XLIM, [bounds_tf(2), bounds_tf(2)], 'Color', bound_color, 'LineStyle', ':', 'LineWidth', 1.2);
    plot(COMPARISONS_XLIM, [bounds_tf(1), bounds_tf(1)], 'Color', bound_color, 'LineStyle', '--', 'LineWidth', 1.2);
end

set(gca, 'XLim', COMPARISONS_XLIM);
set(gca, 'YLim', COMPARISONS_YLIM);
set(gca, 'XTick', [(1:3) - 1/4, (4:6) + 1/4]);
set(gca, 'YGrid', 'on');

set(gca, 'XTickLabel', COMPARISONS_LABELS_CELL);


set(gca, 'TickLabelInterpreter', 'latex');

legend(COMPARISONS_LEGEND, 'Location', 'southwest', 'Interpreter', 'latex');
