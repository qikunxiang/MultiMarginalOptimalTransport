% Compute the lower and upper bounds for the 2-Wasserstein barycenter problem

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

marg_cell = cell(marg_num, 1);

for marg_id = 1:marg_num
    marg_cell{marg_id} = ProbMeas2DCPWADens( ...
        marg_vertices_cell{marg_id}, ...
        marg_triangles_cell{marg_id}, ...
        marg_density_cell{marg_id});
end

input_grid_pts_x = linspace(0, 3, 200 + 1)';
input_grid_pts_y = linspace(0, 3, 200 + 1)';
[input_grid_x, input_grid_y] = meshgrid(input_grid_pts_x, input_grid_pts_y);
input_grid = [input_grid_x(:), input_grid_y(:)];

MMOT_dualfuncs_cell = cell(test_num, 1);

for test_id = 1:test_num
    for marg_id = 1:marg_num
        marg_cell{marg_id}.setSimplicialTestFuncs(marg_testfuncs_cell{test_id}{marg_id}{:});
    end

    OT = MMOT2DWassersteinBarycenter(marg_cell, marg_weights, [], [], []);
    OT.setLSIPSolutions(LSIP_primal_cell{test_id}, LSIP_dual_cell{test_id}, LSIP_UB_list(test_id), LSIP_LB_list(test_id));
    
    vals = OT.evaluateMMOTDualFunctions(input_grid, true);

    % shift the functions by constants such that all but the first function evaluates to 0 at the first input point
    vals(:, 1) = vals(:, 1) + sum(vals(1, 2:end));
    vals(:, 2:end) = vals(:, 2:end) - vals(1, 2:end);

    MMOT_dualfuncs_cell{test_id} = vals;
end

save(CONFIG.SAVEPATH_DUALFUNCS, ...
    'input_grid', ...
    'input_grid_x', ...
    'input_grid_y', ...
    'MMOT_dualfuncs_cell', ...
    '-v7.3');