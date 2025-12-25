% Plot the computed MMOT dual functions

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX = 3;
COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES = 6;
COMPARISONS_OURALGO_TEST_NUM = length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES);

COMPARISONS_OURALGO_LEGEND = cell(COMPARISONS_OURALGO_TEST_NUM * 2, 1);
for test_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    COMPARISONS_OURALGO_LEGEND{COMPARISONS_OURALGO_TEST_NUM * (test_id - 1) + 1} = ...
        ['$\alpha^{\mathsf{UB}}$ ', sprintf('($m_0=%d$)', ...
        testfunc_knot_num_list(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(test_id)))];
    COMPARISONS_OURALGO_LEGEND{COMPARISONS_OURALGO_TEST_NUM * (test_id - 1) + 2} = ...
        ['$\alpha^{\mathsf{LB}}$ ', sprintf('($m_0=%d$)', ...
        testfunc_knot_num_list(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(test_id)))];
end
COMPARISONS_LEGEND = vertcat({'NN-dual'}, COMPARISONS_OURALGO_LEGEND);

COMPARISONS_SAMPLE_NUM = 20;


COMPARISONS_NN_FILEPREFIX = 'NN/';
COMPARISONS_NN_FILESUFFIX_DUAL = '_ReLu_5_64_objectives.txt';

COMPARISONS_NN_FILENAMES_CELL = {
    {'arr1/20_L2_200000', ... 
     'arr1/20_L2_2000000', ...
     'arr1/20_L2_20000000'}, ...
    {'arr2/20_L2_200000', ...
     'arr2/20_L2_2000000', ...
     'arr2/20_L2_20000000'}
};
COMPARISONS_NN_LABELS_CELL = {
    '$\gamma=2\times10^{5}$'; ...
    '$\gamma=2\times10^{6}$'; ...
    '$\gamma=2\times10^{7}$'
};
COMPARISONS_NN_DUAL_COLUMN_NUM = 3;

COMPARISONS_TOTAL_COLUMN_NUM = COMPARISONS_NN_DUAL_COLUMN_NUM;
COMPARISONS_GROUPING = 2 * ones(COMPARISONS_NN_DUAL_COLUMN_NUM, 1);
COMPARISONS_LABELS_CELL = vertcat(COMPARISONS_NN_LABELS_CELL);
COMPARISONS_XLIM = [0.5, COMPARISONS_TOTAL_COLUMN_NUM + 0.4];
COMPARISONS_YLIM = {
    [-1.1, 0.85], ...
    [-5.1, 0.95]
};

bounds_cell = cell(length(arrangements), 1);
obj_cell = cell(length(arrangements), 1);

for arr_id = 1:length(arrangements)
    bounds_cell{arr_id} = zeros(length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES), 2);

    % lower and upper bounds computed by our algorithm
    for tf_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
        test_id = COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(tf_id);
        bounds_cell{arr_id}(tf_id, 1) = OT_LB_cell{arr_id}(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id);
        bounds_cell{arr_id}(tf_id, 2) = OT_UB_cell{arr_id}(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id);
    end

    obj_cell{arr_id} = zeros(COMPARISONS_SAMPLE_NUM, COMPARISONS_TOTAL_COLUMN_NUM);
    col_counter = 0;

    % entropic dual objectives computed by the NN-based algorithm
    for test_id = 1:length(COMPARISONS_NN_FILENAMES_CELL{arr_id})
        txt_filename = [CONFIG.COMPARISONPATH, COMPARISONS_NN_FILEPREFIX, ...
            COMPARISONS_NN_FILENAMES_CELL{arr_id}{test_id}, COMPARISONS_NN_FILESUFFIX_DUAL];

        txt_file = fopen(txt_filename, 'r');
        samps = fscanf(txt_file, '%f');
        fclose(txt_file);

        col_counter = col_counter + 1;
        obj_cell{arr_id}(:, col_counter) = samps;
    end
end


figure('Position', [0, 0, 360, 500]);
ha = tight_subplot(2, 1, [0.006, 0], [0.035, 0.040], [0.140, 0.010]);
for arr_id = 1:length(arrangements)
    axes(ha(arr_id));

    hold on;
    box on;

    BC = boxchart(repelem((1:COMPARISONS_TOTAL_COLUMN_NUM)', COMPARISONS_SAMPLE_NUM, 1), obj_cell{arr_id}(:), ...
        'GroupByColor', repelem(COMPARISONS_GROUPING, COMPARISONS_SAMPLE_NUM, 1), 'BoxWidth', 0.16);

    % change the first group to the second color for consistency between figures
    CO = colororder;
    CO(1, :) = CO(2, :);
    colororder(CO);
    
    for tf_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
        bounds = bounds_cell{arr_id}(tf_id, :);
        
        bound_color = 'red';
        
        plot(COMPARISONS_XLIM, [bounds(2), bounds(2)], 'Color', bound_color, 'LineStyle', ':', 'LineWidth', 1.2);
        plot(COMPARISONS_XLIM, [bounds(1), bounds(1)], 'Color', bound_color, 'LineStyle', '--', 'LineWidth', 1.2);
    end

    set(gca, 'XLim', COMPARISONS_XLIM);
    set(gca, 'YLim', COMPARISONS_YLIM{arr_id});
    set(gca, 'XTick', 1:3);
    set(gca, 'YGrid', 'on');

    if arr_id == 1
        set(gca, 'XTickLabel', []);
        title(sprintf('$N=%d$', timestep_num_list(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX)), 'Interpreter', 'latex', 'FontSize', 16);
    else
        set(gca, 'XTickLabel', COMPARISONS_LABELS_CELL);
    end

    set(gca, 'TickLabelInterpreter', 'latex');
    
    if arr_id == 2
        legend(COMPARISONS_LEGEND, 'Location', 'southwest', 'Interpreter', 'latex');
    end
end

annotation('textbox', [0, 0.31, 0, 0], 'string', '$\Xi^{(2)}$', 'FontSize', 15, 'Interpreter', 'latex')
annotation('textbox', [0, 0.76, 0, 0], 'string', '$\Xi^{(1)}$', 'FontSize', 15, 'Interpreter', 'latex')