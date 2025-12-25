% Perform goodness-of-fit tests on the primal samples produced by the NN-based algorithm

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

COMPARISONS_NN_FILEPREFIX = 'NN/';
COMPARISONS_NN_FILESUFFIX = '_ReLu_5_64.json';

COMPARISONS_NN_FILENAMES_CELL = {
    {'arr1/5_L2_25000', ... 
     'arr1/5_L2_50000', ...
     'arr1/5_L2_500000', ...
     'arr1/5_L2_5000000'}, ...
    {'arr2/5_L2_25000', ...
     'arr2/5_L2_50000', ...
     'arr2/5_L2_500000', ...
     'arr2/5_L2_5000000'}
};

COMPARISONS_NN_REG_LIST = [25000; 50000; 500000; 5000000];

COMPARISONS_NN_TEST_NUM = 4;

output_cell = cell(length(arrangements) * COMPARISONS_NN_TEST_NUM, 5);
counter = 0;

udist = makedist('Uniform');

for arr_id = 1:length(arrangements)
    for test_id = 1:COMPARISONS_NN_TEST_NUM
        counter = counter + 1;

        json_filename = [CONFIG.COMPARISONPATH, COMPARISONS_NN_FILEPREFIX, ...
            COMPARISONS_NN_FILENAMES_CELL{arr_id}{test_id}, COMPARISONS_NN_FILESUFFIX];

        json_data = readstruct(json_filename);

        samps = vertcat(json_data.primal_samples{:});

        marg_num = size(samps, 2);
        pval_list = zeros(marg_num, 1);

        for marg_id = 1:marg_num
            [~, pval_list(marg_id)] = chi2gof(samps(:, marg_id), 'CDF', udist, 'NBins', 10);
        end
        
        output_cell{counter, 1} = arr_id;
        output_cell{counter, 2} = COMPARISONS_NN_REG_LIST(test_id);
        output_cell{counter, 3} = min(pval_list);
        output_cell{counter, 4} = max(pval_list);
        output_cell{counter, 5} = mean(pval_list);
    end
end

strarr = struct('arrangement', output_cell(:, 1), 'reg_param', output_cell(:, 2), ...
    'p_value_min', output_cell(:, 3), 'p_value_max', output_cell(:, 4), 'p_value_avg', output_cell(:, 5));

display(struct2table(strarr));