% Run step 1: computing the lower bounds by solving linear semi-infinite
% programming problems

load('exp/inputs.mat');

LSIP_LB_list = zeros(stepno, 1);
LSIP_UB_list = zeros(stepno, 1);

dual_func_cell = cell(stepno, 1);
coup_meas_cell = cell(stepno, 1);
output_cell = cell(stepno, 1);
options_cell = cell(stepno, 1);

options = struct;
options.display = true;
options.log_file = 'cutplane.log';

options.reduce_cuts = struct;
options.reduce_cuts.thres = 10;
options.reduce_cuts.freq = 500;

options.LP_params = struct;
options.LP_params.OutputFlag = 1;
options.LP_params.LogToConsole = 0;
options.LP_params.LogFile = 'gurobi_LP.log';

options.global_params = struct;
options.global_params.OutputFlag = 1;
options.global_params.LogToConsole = 0;
options.global_params.LogFile = 'gurobi_MILP.log';

options.rescue = struct;
options.rescue.save_file = 'savefile.mat';
options.rescue.save_interval = 100;

for stepid = 1:stepno
    if ~isempty(output_cell{stepid})
        continue;
    end

    if stepid == 1
        % use the comonotone coupling for the first step
        coup_init = coup_init1;
    else
        [atoms_new, prob_new] = reassembly_increment( ...
            knots_cell{stepid - 1}, knots_cell{stepid}, ...
            expval_cell{stepid}, ...
            coup_meas_cell{stepid - 1}.atoms, ...
            coup_meas_cell{stepid - 1}.probs);
        coup_init = unique(atoms_new, 'row');
    end

    [LSIP_UB_list(stepid), LSIP_LB_list(stepid), ...
        coup_meas_cell{stepid}, dual_func_cell{stepid}, ...
        output_cell{stepid}, options_cell{stepid}] ...
        = MMOT_CPWA_cutplane(CPWA, knots_cell{stepid}, ...
        expval_cell{stepid}, coup_init, options);
    
    save('exp/rst_LB.mat', 'LSIP_LB_list', ...
        'LSIP_UB_list', 'dual_func_cell', 'coup_meas_cell', ...
        'output_cell', 'options_cell');
end