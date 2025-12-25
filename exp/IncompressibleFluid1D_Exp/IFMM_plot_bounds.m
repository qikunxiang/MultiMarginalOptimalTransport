% Plot the computed lower and upper bounds

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

xx = testfunc_knot_num_list - 1;
plot_timestep_num_index = 3;

line_width = 1.25;

for arr_id = 1:length(arrangements)

    % figure of lower and upper bounds
    
    figure('Position', [(arr_id - 1) * 800, 100, 400, 250]);
    ha = tight_subplot(1, 1, [0, 0], [0.066, 0.062], [0.095, 0.025]);
    
    hold on;
    
    handle_LB = plot(xx, OT_LB_cell{arr_id}(plot_timestep_num_index, :)', ...
        'Marker', 'o', 'Color', 'blue', ...
        'LineStyle', '--', 'LineWidth', line_width);
    handle_UB = plot(xx, OT_UB_cell{arr_id}(plot_timestep_num_index, :)', ...
        'Marker', 'x', 'Color', 'red', 'LineStyle', ':', ...
        'LineWidth', line_width);
    
    box on;
    grid on;
    
    legend([handle_UB, handle_LB], ...
        {'$\alpha^{\mathsf{UB}}$', ...
        '$\alpha^{\mathsf{LB}}$'}, ...
        'Location', 'northeast', ...
        'Interpreter', 'latex', 'FontSize', 13);
    
    set(gca, 'XScale', 'log');
    set(gca, 'XTick', xx);
    set(gca, 'YLim', [-0.45, 0.15]);
    
    xlabel('number of test functions per marginal');
    ylabel('objective');
    title(sprintf('volume preserving map $\\Xi^{(%d)}$', arr_id), 'Interpreter', 'latex');
    
    % figure of sub-optimalities and error bounds
    
    figure('Position', [400 + (arr_id - 1) * 800, 100, 400, 250]);
    ha = tight_subplot(1, 1, [0, 0], [0.066, 0.062], [0.095, 0.025]);
    
    hold on;
    
    handle_sub = plot(xx, OT_UB_cell{arr_id}(plot_timestep_num_index, :)' - OT_LB_cell{arr_id}(plot_timestep_num_index, :)', ...
        'Marker', 'x', 'Color', 'red', 'LineStyle', '-.', ...
        'LineWidth', line_width);
    handle_th = plot(xx, THEB_cell{arr_id}(plot_timestep_num_index, :)', ...
        'Marker', 'diamond', 'Color', 'black', ...
        'LineStyle', '-', 'LineWidth', line_width);
    
    box on;
    grid on;
    
    legend([handle_sub, handle_th], ...
        {'$\tilde{\epsilon}_{\mathsf{sub}}$', ...
        '$\epsilon_{\mathsf{theo}}$'}, ...
        'Location', 'northeast', ...
        'Interpreter', 'latex', 'FontSize', 13);
    
    set(gca, 'XTick', xx);
    set(gca, 'XScale', 'log');
    set(gca, 'YScale', 'log');
    set(gca, 'YLim', [1e-4, 1e2]);
    
    xlabel('number of test functions per marginal');
    ylabel('sub-optimality');
    title(sprintf('volume preserving map $\\Xi^{(%d)}$', arr_id), 'Interpreter', 'latex');

end