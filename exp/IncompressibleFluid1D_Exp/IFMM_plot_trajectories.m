% Plot the particle trajectories at intermediate time steps

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

plot_timestep_num_index = 9;
plot_testfunc_num_index = 6;
timestep_num = timestep_num_list(plot_timestep_num_index);

xx = linspace(0, 1, 200)';
tt = (0:timestep_num)' / timestep_num;

color_list = jet(length(xx));

for arr_id = 1:length(arrangements)

    OT = MMOT1DFluid(timestep_num_list(plot_timestep_num_index), ...
        arrangements{arr_id}, ...
        testfunc_knot_num_list(plot_testfunc_num_index));

    OT.setLSIPSolutions(LSIP_primal_cell{arr_id}{ ...
        plot_timestep_num_index, plot_testfunc_num_index}, ...
        LSIP_dual_cell{arr_id}{plot_timestep_num_index, ...
        plot_testfunc_num_index}, ...
        LSIP_UB_cell{arr_id}(plot_timestep_num_index, ...
        plot_testfunc_num_index), ...
        LSIP_LB_cell{arr_id}(plot_timestep_num_index, ...
        plot_testfunc_num_index));
    OT.setComonotoneMap(OT_comonotone_map_cell{arr_id}{ ...
        plot_timestep_num_index, plot_testfunc_num_index});

    map_eval = OT.evaluateOptMongeMaps(xx);


    figure('Position', [400 * arr_id, 100, 400, 400]);
    ha = tight_subplot(1, 1, [0, 0], ...
        [0.08, 0.04], [0.09, 0.015]);
    axes(ha(1));
    hold on;
    box on;

    for xx_id = 1:length(xx)
        plot(map_eval(xx_id, :), tt, 'Color', color_list(xx_id, :));
    end

    set(gca, 'XLim', [0, 1]);
    set(gca, 'YLim', [0, 1]);
    
    xlabel('position');
    ylabel('$t$', 'Interpreter', 'latex');

    title(sprintf('volume preserving map $\\Xi^{(%d)}$', ...
        arr_id), 'Interpreter', 'latex');
end