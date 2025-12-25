% Plot the computed MMOT dual functions

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX = 2;
COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES = [4, 5, 6];

COMPARISONS_OURALGO_TITLES_CELL = cell(length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES), 1);
for test_id = 1:length(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES)
    COMPARISONS_OURALGO_TITLES_CELL{test_id} = sprintf('\\textbf{our algo.} ($m_0=%d$)', ...
        testfunc_knot_num_list(COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES(test_id)));
end

COMPARISONS_OURALGO_COLUMN_NUM = 3;

COMPARISONS_RKHS_FILEPREFIX = 'RKHS/';
COMPARISONS_RKHS_FILESUFFIX = '_Laplace_0.5_exp.json';

COMPARISONS_RKHS_FILENAMES_CELL = {
    {'arr1/10_L2_25000', ... 
     'arr1/10_L2_50000'}, ...
    {'arr2/10_L2_25000', ...
     'arr2/10_L2_50000'}
};
COMPARISONS_RKHS_TITLES_CELL = {
    'RKHS ($\gamma=2.5\times 10^{4}$)'; ...
    'RKHS ($\gamma=5.0\times 10^{4}$)'
};
COMPARISONS_RKHS_COLUMN_NUM = 2;

COMPARISONS_NN_FILEPREFIX = 'NN/';
COMPARISONS_NN_FILESUFFIX = '_ReLu_5_64.json';

COMPARISONS_NN_FILENAMES_CELL = {
    {'arr1/10_L2_100000', ... 
     'arr1/10_L2_1000000', ...
     'arr1/10_L2_10000000'}, ...
    {'arr2/10_L2_100000', ...
     'arr2/10_L2_1000000', ...
     'arr2/10_L2_10000000'}
};
COMPARISONS_NN_TITLES_CELL = {
    'NN ($\gamma=10^{5}$)'; ...
    'NN ($\gamma=10^{6}$)'; ...
    'NN ($\gamma=10^{7}$)'
};
COMPARISONS_NN_COLUMN_NUM = 3;

COMPARISONS_TOTAL_COLUMN_NUM = COMPARISONS_OURALGO_COLUMN_NUM + COMPARISONS_RKHS_COLUMN_NUM + COMPARISONS_NN_COLUMN_NUM;
COMPARISONS_TITLES_CELL = vertcat(COMPARISONS_OURALGO_TITLES_CELL, COMPARISONS_RKHS_TITLES_CELL, COMPARISONS_NN_TITLES_CELL);

YLIM_CELL = {
    [-0.05, 0.6], ... 
    [-0.05, 0.6]
};

funcs_cell = cell(COMPARISONS_TOTAL_COLUMN_NUM, length(arrangements));

for arr_id = 1:length(arrangements)
    col_counter = 0;

    % dual functions computed by our algorithm
    for test_id = COMPARISONS_OURALGO_TESTFUNC_NUM_INDICES
        func = struct;
        func.inputs = linspace(0, 1, 400 + 1)';

        OT = MMOT1DFluid(timestep_num_list(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX), arrangements{arr_id}, ...
            testfunc_knot_num_list(test_id));

        OT.setLSIPSolutions(LSIP_primal_cell{arr_id}{COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id}, ...
            LSIP_dual_cell{arr_id}{COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id}, ...
            LSIP_UB_cell{arr_id}(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id), ...
            LSIP_LB_cell{arr_id}(COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id));
        OT.setComonotoneMap(OT_comonotone_map_cell{arr_id}{COMPARISONS_OURALGO_TIMESTEP_NUM_INDEX, test_id});

        func.values = OT.evaluateMMOTDualFunctions(func.inputs, true);
        col_counter = col_counter + 1;
        funcs_cell{col_counter, arr_id} = func;
    end

    % dual functions computed by the RKHS-based algorithm
    for test_id = 1:length(COMPARISONS_RKHS_FILENAMES_CELL{arr_id})
        json_filename = [CONFIG.COMPARISONPATH, COMPARISONS_RKHS_FILEPREFIX, ...
            COMPARISONS_RKHS_FILENAMES_CELL{arr_id}{test_id}, COMPARISONS_RKHS_FILESUFFIX];

        json_data = readstruct(json_filename);
        func = struct;
        func.inputs = json_data.dual_funcs_eval_points';
        func.values = vertcat(json_data.dual_funcs_vals{:})';
        col_counter = col_counter + 1;
        funcs_cell{col_counter, arr_id} = func;
    end

    % dual functions computed by the NN-based algorithm
    for test_id = 1:length(COMPARISONS_NN_FILENAMES_CELL{arr_id})
        json_filename = [CONFIG.COMPARISONPATH, COMPARISONS_NN_FILEPREFIX, ...
            COMPARISONS_NN_FILENAMES_CELL{arr_id}{test_id}, COMPARISONS_NN_FILESUFFIX];

        json_data = readstruct(json_filename);
        func = struct;
        func.inputs = json_data.dual_funcs_eval_points';
        func.values = vertcat(json_data.dual_funcs_vals{:})';
        col_counter = col_counter + 1;
        funcs_cell{col_counter, arr_id} = func;
    end
end

figure('Position', [0, 0, 1500, 400]);
ha = tight_subplot(length(arrangements), COMPARISONS_TOTAL_COLUMN_NUM, [0.010, 0.003], [0.010, 0.048], [0.020, 0.003]);

for arr_id = 1:length(arrangements)
    for col_id = 1:COMPARISONS_TOTAL_COLUMN_NUM
        func = funcs_cell{col_id, arr_id};
        vals = func.values;
        
        % shift the functions by constants such that all but the first function evaluates to 0 at 0
        vals(:, 1) = vals(:, 1) + sum(vals(1, 2:end));
        vals(:, 2:end) = vals(:, 2:end) - vals(1, 2:end);

        axes(ha((arr_id - 1) * COMPARISONS_TOTAL_COLUMN_NUM + col_id));

        plot(func.inputs, vals);
        set(gca, 'XTickLabels', []);
        set(gca, 'YTickLabels', []);
        set(gca, 'YLim', YLIM_CELL{arr_id});
        box on;
        grid on;

        if arr_id == 1
            title(COMPARISONS_TITLES_CELL{col_id}, 'Interpreter', 'latex', 'FontSize', 14)
        end
    end
end

annotation('textbox', [0, 0.27, 0, 0], 'string', '$\Xi^{(2)}$', 'FontSize', 15, 'Interpreter', 'latex')
annotation('textbox', [0, 0.73, 0, 0], 'string', '$\Xi^{(1)}$', 'FontSize', 15, 'Interpreter', 'latex')