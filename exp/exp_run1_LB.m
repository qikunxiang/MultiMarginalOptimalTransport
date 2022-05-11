load('exp/exp_inputs.mat');

LSIP_LB_list = zeros(step_no, 1);
LSIP_UB_list = zeros(step_no, 1);

dual_func_cell = cell(step_no, 1);
coup_meas_cell = cell(step_no, 1);
output_cell = cell(step_no, 1);
options_cell = cell(step_no, 1);

options = struct;
options.display = true;
options.log_file = 'exp_cutplane.log';

options.reduce_cuts = struct;
options.reduce_cuts.thres = 10;
options.reduce_cuts.freq = 500;

options.LP_params = struct;
options.LP_params.OutputFlag = 1;
options.LP_params.LogToConsole = 0;
options.LP_params.LogFile = 'exp_gurobi_LP.log';

options.global_params = struct;
options.global_params.OutputFlag = 1;
options.global_params.LogToConsole = 0;
options.global_params.LogFile = 'exp_gurobi_MILP.log';

options.rescue = struct;
options.rescue.save_file = 'exp_savefile.mat';
options.rescue.save_interval = 100;

for step_id = 1:step_no
    if ~isempty(output_cell{step_id})
        continue;
    end

    if step_id == 1
        % use the comonotone coupling for the first step
        coup_init = coup_init1;
    else
        [atoms_new, probs_new] = reassembly_increment( ...
            knots_cell{step_id}, expval_cell{step_id}, ...
            coup_meas_cell{step_id - 1}.atoms, ...
            coup_meas_cell{step_id - 1}.probs);
        coup_init = unique(atoms_new, 'row');
    end

    [LSIP_UB_list(step_id), LSIP_LB_list(step_id), ...
        coup_meas_cell{step_id}, dual_func_cell{step_id}, ...
        output_cell{step_id}, options_cell{step_id}] ...
        = MMOT_CPWA_cutplane(CPWA, knots_cell{step_id}, ...
        expval_cell{step_id}, coup_init, options);
    
    save('exp/exp_rst_LB.mat', 'LSIP_LB_list', ...
        'LSIP_UB_list', 'dual_func_cell', 'coup_meas_cell', ...
        'output_cell', 'options_cell');
end