% Compute the lower and upper bounds for the 2-Wasserstein barycenter
% problem

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_W2OT);

options = struct;
options.log_file = CONFIG.LOGPATH_W2OTUB;
options.display = true;

MCsamp_num = 1e7;
MCrep_num = 1000;

MMOT_W2OTUB_cell = cell(test_num, 1);
MMOT_W2OTUB_mean_list = zeros(test_num, 1);
MMOT_W2OTdiff_cell = cell(test_num, 1);
MMOT_W2OTdiff_mean_list = zeros(test_num, 1);
WB_W2OT_histpdf_cell = cell(test_num, 1);


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
    for marg_id = 1:marg_num
        marg_cell{marg_id}.setSimplicialTestFuncs( ...
            marg_testfuncs_cell{test_id}{marg_id}{:});
    end

    OT = MMOT2DWassersteinBarycenter(marg_cell, marg_weights, ...
        options, [], []);
    OT.setLSIPSolutions(LSIP_primal_cell{test_id}, ...
        LSIP_dual_cell{test_id}, ...
        LSIP_UB_list(test_id), LSIP_LB_list(test_id));
    OT.loadW2OptimalTransportInfo(W2OT_info_cell{test_id, 1}, ...
        W2OT_info_cell{test_id, 2});
    
    MMOT_LB = OT.getMMOTLowerBound();

    RS = RandStream('mrg32k3a', 'Seed', 3000);
    RS.Substream = test_id;
    [MMOT_W2OTUB_list, MCsamp, WB_histpdf] ...
        = OT.getMMOTUpperBoundViaW2CouplingWRepetition( ...
        MCsamp_num, MCrep_num, RS, ...
        WB_hist_edge_x, WB_hist_edge_y);
    MMOT_W2OTUB_mean = mean(MMOT_W2OTUB_list);
    MMOT_W2OTdiff_list = MMOT_W2OTUB_list - MMOT_LB;
    MMOT_W2OTdiff_mean = mean(MMOT_W2OTdiff_list);

    log_text = sprintf(['test %2d: LB = %10.4f, UB = %10.4f, ' ...
        'diff = %10.6f\n'], test_id, ...
        MMOT_LB, MMOT_W2OTUB_mean, MMOT_W2OTdiff_mean);

    fprintf(main_log_file, log_text);
    fprintf(log_text);

    MMOT_W2OTUB_cell{test_id} = MMOT_W2OTUB_list;
    MMOT_W2OTUB_mean_list(test_id) = MMOT_W2OTUB_mean;
    MMOT_W2OTdiff_cell{test_id} = MMOT_W2OTdiff_list;
    MMOT_W2OTdiff_mean_list(test_id) = MMOT_W2OTdiff_mean;
    WB_W2OT_histpdf_cell{test_id} = WB_histpdf;

    save(CONFIG.SAVEPATH_W2OTUB, ...
        'MMOT_W2OTUB_cell', ...
        'MMOT_W2OTUB_mean_list', ...
        'MMOT_W2OTdiff_cell', ...
        'MMOT_W2OTdiff_mean_list', ...
        'WB_W2OT_histpdf_cell', ...
        '-v7.3');
end

fprintf(main_log_file, '--- experiment ends ---\n\n');
fclose(main_log_file);