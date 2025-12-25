% Plot the computed lower and upper bounds

CONFIG = CPWAMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

xx = zeros(test_num, 1);

for test_id = 1:test_num
    xx(test_id) = size(testfuncs_cell{test_id}{1}{1}, 1) - 1;
end

OT_UB_err = zeros(test_num, 2);

for test_id = 1:test_num
    qtt = quantile(OT_UB_samps_cell{test_id, 1}, [0.025, 0.975]);

    OT_UB_err(test_id, 1) = OT_UB_mean_list(test_id) - qtt(1);
    OT_UB_err(test_id, 2) = qtt(2) - OT_UB_mean_list(test_id);
end

line_width = 1.25;

% figure of lower and upper bounds

figure('Position', [0, 100, 400, 250]);
ha = tight_subplot(1, 1, [0, 0], [0.066, 0.018], [0.105, 0.025]);
axes(ha(1));

hold on;

handle_LB = plot(xx, OT_LB_list, 'Marker', 'o', 'Color', 'blue', ...
    'LineStyle', '--', 'LineWidth', line_width);
handle_UB = errorbar(xx, OT_UB_mean_list, ...
    OT_UB_err(:, 1), OT_UB_err(:, 2), ...
    'Marker', 'none', 'Color', 'red', 'LineStyle', ':', ...
    'LineWidth', line_width);

box on;
grid on;

legend([handle_UB, handle_LB], ...
    {'$\alpha^{\mathsf{UB}}$', ...
    '$\alpha^{\mathsf{LB}}$'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 13);

set(gca, 'XTick', xx);
set(gca, 'XLim', [min(xx), max(xx)]);
set(gca, 'XScale', 'log');

xlabel('number of test functions per marginal');
ylabel('objective');


% figure of sub-optimalities and error bounds

figure('Position', [400, 100, 400, 250]);
ha = tight_subplot(1, 1, [0, 0], [0.066, 0.018], [0.105, 0.025]);
axes(ha(1));

hold on;

handle_sub = errorbar(xx, OT_UB_mean_list - OT_LB_list, ...
    OT_UB_err(:, 1), OT_UB_err(:, 2), ...
    'Marker', 'none', 'Color', 'red', 'LineStyle', ':', 'LineWidth', line_width);
handle_th = plot(xx, THEB_list, ...
    'Marker', 'diamond', 'Color', 'black', 'LineStyle', '-', 'LineWidth', line_width);

box on;
grid on;

legend([handle_sub, handle_th], ...
    {'$\tilde{\epsilon}_{\mathsf{sub}}$', ...
    '$\epsilon_{\mathsf{theo}}$'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 13);

set(gca, 'XTick', xx);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
set(gca, 'XLim', [min(xx), max(xx)]);
set(gca, 'YLim', [1e-1, 5e3]);

xlabel('number of test functions per marginal');
ylabel('sub-optimality');
