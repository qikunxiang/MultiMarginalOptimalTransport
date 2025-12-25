% Plot the volume preserving maps at intermediate time steps

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

plot_timestep_num_index = 3;
plot_testfunc_num_index = 6;
timestep_num = timestep_num_list(plot_timestep_num_index);

xx = linspace(0, 1, 1000)';

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
    

    figure('Position', [200, 100, 1200, 580]);
    ha = tight_subplot(3, 7, [0.035, 0.004], ...
        [0.035, 0.0035], [0.003, 0.003]);

    for ts_id = 1:timestep_num + 1
        axes(ha(ts_id));
        
        plot(xx, map_eval(:, ts_id), 'Color', 'black');
        set(gca, 'XTickLabels', []);
        set(gca, 'YTickLabels', []);
        box on;
        grid on;

        if ts_id == 1
            xlabel('$t=0$', 'Interpreter', 'latex');
        elseif ts_id == timestep_num + 1
            xlabel('$t=1$', 'Interpreter', 'latex');
        else
            xlabel(sprintf('$t=%d/%d$', ts_id - 1, timestep_num), ...
                'Interpreter', 'latex');
        end
        
    end
end