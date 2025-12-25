% Incompressible fluid example with 1D marginals

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);

options = struct;
options.log_file = CONFIG.LOGPATH_LSIP_MAIN;
options.display = true;
options.reduce = struct;
options.reduce.preserve_init_constr = true;
options.reduce.min_slack = 1e-7;
options.reduce.thres = 5e-3;
options.reduce.thres_quantile = 0.6;
options.reduce.max_iter = 15000;
options.reduce.freq = 500;

global_options = struct;
global_options.pool_size = 10;

LP_options = struct;
LP_options.OutputFlag = 1;
LP_options.LogToConsole = 1;
LP_options.LogFile = CONFIG.LOGPATH_LSIP_LP;

arrangment_num = length(arrangements);
timestep_test_num = length(timestep_num_list);
testfunc_test_num = length(testfunc_knot_num_list);

tolerance = 1e-5;

output_cell = cell(arrangment_num, 1);
OT_LB_cell = cell(arrangment_num, 1);
OT_UB_cell = cell(arrangment_num, 1);
OT_err_cell = cell(arrangment_num, 1);
THEB_cell = cell(arrangment_num, 1);
LSIP_LB_cell = cell(arrangment_num, 1);
LSIP_UB_cell = cell(arrangment_num, 1);
LSIP_primal_cell = cell(arrangment_num, 1);
LSIP_dual_cell = cell(arrangment_num, 1);
OT_comonotone_map_cell = cell(arrangment_num, 1);

main_log_file = fopen(CONFIG.LOGPATH_MAIN, 'a');

if main_log_file < 0
    error('cannot open log file');
end

fprintf(main_log_file, '--- experiment starts ---\n');

for arr_id = 1:arrangment_num

    if arr_id > 1
        OT_LB_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        OT_UB_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        OT_err_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        THEB_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        
        LSIP_LB_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        LSIP_UB_cell{arr_id} = zeros(timestep_test_num, testfunc_test_num);
        LSIP_primal_cell{arr_id} = cell(timestep_test_num, testfunc_test_num);
        LSIP_dual_cell{arr_id} = cell(timestep_test_num, testfunc_test_num);
        OT_comonotone_map_cell{arr_id} = ...
            cell(timestep_test_num, testfunc_test_num);
        
        output_cell{arr_id} = cell(timestep_test_num, testfunc_test_num);
    end
    
    
    for timestep_id = 1:timestep_test_num
        for testfunc_id = 1:testfunc_test_num

            if testfunc_id == 1
                OT = MMOT1DFluid(timestep_num_list(timestep_id), arrangements{arr_id}, testfunc_knot_num_list(testfunc_id), ...
                    options, LP_options, global_options);
    
                coup_indices = OT.generateHeuristicCoupling();
            else
                coup_indices = OT.updateSimplicialTestFuncs(testfunc_knot_num_list(testfunc_id));
            end
            
            init_constr = struct('knot_indices', coup_indices);
            
            output_cell{arr_id}{timestep_id, testfunc_id} = OT.run(init_constr, tolerance);
            
            OT_LB = OT.getMMOTLowerBound();
            OT_UB = OT.getMMOTUpperBound();
            THEB = OT.getMTTheoreticalErrorBound(tolerance);
        
            OT_LB_cell{arr_id}(timestep_id, testfunc_id) = OT_LB;
            OT_UB_cell{arr_id}(timestep_id, testfunc_id) = OT_UB;
            OT_err_cell{arr_id}(timestep_id, testfunc_id) = OT_UB - OT_LB;
            THEB_cell{arr_id}(timestep_id, testfunc_id) = THEB;
        
            LSIP_LB_cell{arr_id}(timestep_id, testfunc_id) = OT.Runtime.LSIP_LB;
            LSIP_UB_cell{arr_id}(timestep_id, testfunc_id) = OT.Runtime.LSIP_UB;
            LSIP_primal_cell{arr_id}{timestep_id, testfunc_id} = OT.Runtime.PrimalSolution;
            LSIP_dual_cell{arr_id}{timestep_id, testfunc_id} = OT.Runtime.DualSolution;
            OT_comonotone_map_cell{arr_id}{timestep_id, testfunc_id} = OT.Runtime.ComonotoneMap;
        
            log_text = sprintf(['arrangement %1d, %2d time steps, %3d knots: ' ...
                'LB = %8.4f, UB = %8.4f, diff = %10.6f, THEB = %10.6f\n'], ...
                arr_id, timestep_num_list(timestep_id), testfunc_knot_num_list(testfunc_id), ...
                OT_LB, OT_UB, OT_UB - OT_LB, THEB);
        
            fprintf(main_log_file, log_text);
            fprintf(log_text);
        
            save(CONFIG.SAVEPATH_OUTPUTS, ...
                'output_cell', ...
                'LSIP_LB_cell', ...
                'LSIP_UB_cell', ...
                'LSIP_primal_cell', ...
                'LSIP_dual_cell', ...
                'OT_comonotone_map_cell', ...
                'OT_LB_cell', ...
                'OT_UB_cell', ...
                'OT_err_cell', ...
                'THEB_cell', ...
                '-v7.3');
        end
    end
end

fprintf(main_log_file, '--- experiment ends ---\n\n');
fclose(main_log_file);