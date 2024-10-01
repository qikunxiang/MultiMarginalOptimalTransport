% Example of multi-marginal optimal transport (MMOT) problem with
% 1D marginals and a continuous piece-wise affine cost function

CONFIG = CPWAMM_config();

load(CONFIG.SAVEPATH_INPUTS);

options = struct;
options.global_formulation = 'LOG';
options.log_file = CONFIG.LOGPATH_LSIP_MAIN;
options.display = true;
options.reduce = struct;
options.reduce.thres = 5e-2;
options.reduce.max_iter = 8000;
options.reduce.freq = 50;

global_options = struct;
global_options.PoolSolutions = 100;
global_options.OutputFlag = 1;
global_options.LogToConsole = 1;
global_options.LogFile = CONFIG.LOGPATH_LSIP_GLOBAL;

LP_options = struct;
LP_options.OutputFlag = 1;
LP_options.LogToConsole = 1;
LP_options.LogFile = CONFIG.LOGPATH_LSIP_LP;

tolerance = 1e-4;

MCsamp_num = 1e6;
MCrep_num = 100;

output_cell = cell(test_num, 1);
OT_LB_list = zeros(test_num, 1);
OT_UB_samps_cell = cell(test_num, 1);
OT_UB_mean_list = zeros(test_num, 1);
OT_err_list = zeros(test_num, 1);
THEB_list = zeros(test_num, 1);
LSIP_LB_list = zeros(test_num, 1);
LSIP_UB_list = zeros(test_num, 1);
LSIP_primal_cell = cell(test_num, 1);
LSIP_dual_cell = cell(test_num, 1);

main_log_file = fopen(CONFIG.LOGPATH_MAIN, 'a');

if main_log_file < 0
    error('cannot open log file');
end

fprintf(main_log_file, '--- experiment starts ---\n');

for test_id = 1:test_num
    if test_id == 1
        marg_cell = cell(marg_num, 1);

        for marg_id = 1:marg_num
            marg_cell{marg_id} = ...
                ProbMeas1DMixNorm(marg_params_cell{marg_id}{:});
            marg_cell{marg_id}.setSimplicialTestFuncs( ...
                testfuncs_cell{test_id}{marg_id}{:});
        end

        OT = MMOT1DCPWA(marg_cell, costfunc, ...
            options, LP_options, global_options);

        [coup_atoms, testfunc_vals] = OT.generateHeuristicCoupling();
    else
        [coup_atoms, testfunc_vals] = OT.updateSimplicialTestFuncs( ...
            testfuncs_cell{test_id});
    end

    init_constr = struct('points', coup_atoms, ...
        'testfunc_vals', testfunc_vals);

    output_cell{test_id} = OT.run(init_constr, tolerance);

    OT_LB = OT.getMMOTLowerBound();

    RS = RandStream("combRecursive", 'Seed', 1000 + test_id * 100);
    [OT_UB_list, samps] = OT.getMMOTUpperBoundWRepetition( ...
        MCsamp_num, MCrep_num, RS);
    OT_UB_mean = mean(OT_UB_list);
    OTEB = OT.getMTErrorBoundBasedOnOT(tolerance);
    THEB = OT.getMTTheoreticalErrorBound(tolerance);

    OT_LB_list(test_id) = OT_LB;
    OT_UB_samps_cell{test_id} = OT_UB_list;
    OT_UB_mean_list(test_id) = OT_UB_mean;
    OT_err_list(test_id) = OT_UB_mean - OT_LB;
    THEB_list(test_id) = THEB;

    LSIP_LB_list(test_id) = OT.Runtime.LSIP_LB;
    LSIP_UB_list(test_id) = OT.Runtime.LSIP_UB;
    LSIP_primal_cell{test_id} = OT.Runtime.PrimalSolution;
    LSIP_dual_cell{test_id} = OT.Runtime.DualSolution;

    log_text = sprintf(['test %2d: LB = %10.4f, UB = %10.4f, ' ...
        'diff = %10.6f, ' ...
        'OTEB = %10.6f, THEB = %10.6f\n'], test_id, ...
        OT_LB, OT_UB_mean, OT_UB_mean - OT_LB, ...
        OTEB, THEB);

    fprintf(main_log_file, log_text);
    fprintf(log_text);

    save(CONFIG.SAVEPATH_OUTPUTS, ...
        'output_cell', ...
        'LSIP_LB_list', ...
        'LSIP_UB_list', ...
        'LSIP_primal_cell', ...
        'LSIP_dual_cell', ...
        'OT_LB_list', ...
        'OT_UB_samps_cell', ...
        'OT_UB_mean_list', ...
        'OT_err_list', ...
        'THEB_list', ...
        '-v7.3');
end

fprintf(main_log_file, '--- experiment ends ---\n\n');
fclose(main_log_file);