% Compute the lower and upper bounds for the 2-Wasserstein barycenter
% problem

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OT);

options = struct;
options.log_file = CONFIG.LOGPATH_LSIP_MAIN;
options.display = true;
options.reduce = struct;
options.reduce.thres = 5e-2;
options.reduce.max_iter = 8000;
options.reduce.freq = 50;
options.OT = struct;
options.OT.optimization_options = struct;
options.OT.optimization_options.Display = 'iter-detailed';

global_options = struct;
global_options.pool_size = 100;
global_options.log_file = CONFIG.LOGPATH_LSIP_GLOBAL;

LP_options = struct;
LP_options.OutputFlag = 1;
LP_options.LogToConsole = 1;
LP_options.LogFile = CONFIG.LOGPATH_LSIP_LP;

tolerance = 1e-4;

output_cell = cell(test_num, 1);
LSIP_primal_cell = cell(test_num, 1);
LSIP_dual_cell = cell(test_num, 1);
LSIP_LB_list = zeros(test_num, 1);
LSIP_UB_list = zeros(test_num, 1);


main_log_file = fopen(CONFIG.LOGPATH_MAIN, 'a');

if main_log_file < 0
    error('cannot open log file');
end

fprintf(main_log_file, '--- experiment starts ---\n');

marg_cell = cell(marg_num, 1);

for marg_id = 1:marg_num
    marg_cell{marg_id} = ProbMeas2DCPWADens( ...
        marg_vertices_cell{marg_id}, ...
        marg_triangles_cell{marg_id}, ...
        marg_density_cell{marg_id});
end


for test_id = 1:test_num

    if test_id == 1
        for marg_id = 1:marg_num
            marg_cell{marg_id}.setSimplicialTestFuncs( ...
                marg_testfuncs_cell{test_id}{marg_id}{:});
        end
    
        OT = MMOT2DWassersteinBarycenter(marg_cell, marg_weights, ...
            options, LP_options, global_options);
    
        coup_indices = OT.generateHeuristicCoupling();
    else
        coup_indices = ...
            OT.updateSimplicialTestFuncs(marg_testfuncs_cell{test_id});
    end

    initial_constr = struct('vertex_indices', coup_indices);

    output = OT.run(initial_constr, tolerance);

    OT.loadOptimalTransportInfo(OT_info_cell{test_id});

    LSIP_primal = OT.Runtime.PrimalSolution;
    LSIP_dual = OT.Runtime.DualSolution;
    LSIP_LB = OT.Runtime.LSIP_LB;
    LSIP_UB = OT.Runtime.LSIP_UB;

    log_text = sprintf('test %2d: LB = %10.4f\n', test_id, MMOT_LB);

    fprintf(main_log_file, log_text);
    fprintf(log_text);

    output_cell{test_id} = output;
    LSIP_primal_cell{test_id} = LSIP_primal;
    LSIP_dual_cell{test_id} = LSIP_dual;
    LSIP_LB_list(test_id) = LSIP_LB;
    LSIP_UB_list(test_id) = LSIP_UB;

    save(CONFIG.SAVEPATH_OUTPUTS, ...
        'tolerance', ...
        'output_cell', ...
        'LSIP_primal_cell', ...
        'LSIP_dual_cell', ...
        'LSIP_LB_list', ...
        'LSIP_UB_list', ...
        '-v7.3');
end

fprintf(main_log_file, '--- experiment ends ---\n\n');
fclose(main_log_file);