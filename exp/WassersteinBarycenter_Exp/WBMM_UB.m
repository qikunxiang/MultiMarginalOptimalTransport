% Compute the lower and upper bounds for the 2-Wasserstein barycenter
% problem

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_OT);

options = struct;
options.log_file = CONFIG.LOGPATH_UB;
options.display = true;

MCsamp_num = 1e7;
MCrep_num = 1000;

MMOT_LB_list = zeros(test_num, 1);
MMOT_UB_cell = cell(test_num, 1);
MMOT_UB_mean_list = zeros(test_num, 1);
MMOT_diff_cell = cell(test_num, 1);
MMOT_diff_mean_list = zeros(test_num, 1);
MMOT_OTEB_list = zeros(test_num, 1);
MMOT_THEB_list = zeros(test_num, 1);
WB_histpdf_cell = cell(test_num, 1);


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
    OT.loadOptimalTransportInfo(OT_info_cell{test_id});

    MMOT_LB = OT.getMMOTLowerBound();

    RS = RandStream('mrg32k3a', 'Seed', 1000);
    RS.Substream = test_id;
    [MMOT_UB_list, MCsamp, WB_histpdf] ...
        = OT.getMMOTUpperBoundWRepetition( ...
        MCsamp_num, MCrep_num, RS, ...
        WB_hist_edge_x, WB_hist_edge_y);
    MMOT_UB_mean = mean(MMOT_UB_list);
    MMOT_diff_list = MMOT_UB_list - MMOT_LB;
    MMOT_diff_mean = mean(MMOT_diff_list);

    MMOT_OTEB = OT.getMMOTErrorBoundBasedOnOT(tolerance);
    MMOT_THEB = OT.getMMOTTheoreticalErrorBound(tolerance);

    log_text = sprintf(['test %2d: LB = %10.4f, UB = %10.4f, ' ...
        'diff = %10.6f, ' ...
        'OTEB = %10.6f, THEB = %10.6f\n'], test_id, ...
        MMOT_LB, MMOT_UB_mean, MMOT_diff_mean, ...
        MMOT_OTEB, MMOT_THEB);

    fprintf(main_log_file, log_text);
    fprintf(log_text);

    MMOT_LB_list(test_id) = MMOT_LB;
    MMOT_UB_cell{test_id} = MMOT_UB_list;
    MMOT_UB_mean_list(test_id) = MMOT_UB_mean;
    MMOT_diff_cell{test_id} = MMOT_diff_list;
    MMOT_diff_mean_list(test_id) = MMOT_diff_mean;
    MMOT_OTEB_list(test_id) = MMOT_OTEB;
    MMOT_THEB_list(test_id) = MMOT_THEB;
    WB_histpdf_cell{test_id} = WB_histpdf;

    save(CONFIG.SAVEPATH_UB, ...
        'MMOT_LB_list', ...
        'MMOT_UB_cell', ...
        'MMOT_UB_mean_list', ...
        'MMOT_diff_cell', ...
        'MMOT_diff_mean_list', ...
        'MMOT_OTEB_list', ...
        'MMOT_THEB_list', ...
        'WB_histpdf_cell', ...
        '-v7.3');
end

fprintf(main_log_file, '--- experiment ends ---\n\n');
fclose(main_log_file);