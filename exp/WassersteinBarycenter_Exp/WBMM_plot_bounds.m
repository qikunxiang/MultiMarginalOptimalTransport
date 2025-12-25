% Plot the computed lower and upper bounds

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_UB);
load(CONFIG.SAVEPATH_W2OTUB);

xx = zeros(test_num, 1);

for test_id = 1:test_num
    xx(test_id) = size(marg_testfuncs_cell{test_id}{1}{1}, 1) - 1;
end

MMOT_UB_err = zeros(test_num, 2);

% since we are generating 10^7 samples for 1000 repetitions instead of generating 10^8 samples for 100 repetitions, we need to first 
% regroup the repetitions and compute their mean values

for test_id = 1:test_num
    MMOT_UB_processed = mean(reshape(MMOT_UB_cell{test_id, 1}, [], 10), 2);
    qtt = quantile(MMOT_UB_processed, [0.025, 0.975]);

    MMOT_UB_err(test_id, 1) = MMOT_UB_mean_list(test_id) - qtt(1);
    MMOT_UB_err(test_id, 2) = qtt(2) - MMOT_UB_mean_list(test_id);
end

MMOT_W2OTUB_err = zeros(test_num, 2);

for test_id = 1:test_num
    MMOT_W2OTUB_processed = mean(reshape(MMOT_W2OTUB_cell{test_id, 1}, ...
        [], 10), 2);
    qtt = quantile(MMOT_W2OTUB_processed, [0.025, 0.975]);

    MMOT_W2OTUB_err(test_id, 1) = MMOT_W2OTUB_mean_list(test_id) - qtt(1);
    MMOT_W2OTUB_err(test_id, 2) = qtt(2) - MMOT_W2OTUB_mean_list(test_id);
end

line_width = 1.25;

% figure of all lower and upper bounds

figure('Position', [0, 100, 400, 300]);
ha = tight_subplot(1, 1, [0, 0], [0.135, 0.020], [0.105, 0.025]);
axes(ha(1));

hold on;

handle_LB = plot(xx, MMOT_LB_list, ...
    'Marker', 'o', 'Color', 'blue', 'LineStyle', '--', 'LineWidth', line_width);
handle_UB = plot(xx, MMOT_UB_mean_list, ...
    'Marker', 'x', 'Color', 'red', 'LineStyle', ':', 'LineWidth', line_width);
handle_W2OTUB = plot(xx, MMOT_W2OTUB_mean_list, ...
    'Marker', '+', 'Color', 'green', 'LineStyle', '-.', 'LineWidth', line_width);

box on;
grid on;

legend([handle_UB, handle_W2OTUB, handle_LB], ...
    {'$\alpha^{\mathsf{UB}}$', ...
    '$\beta^{\mathsf{UB}}$', ...
    '$\alpha^{\mathsf{LB}}$'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 13);

set(gca, 'XTick', xx);
xtickangle(25);
set(gca, 'XLim', [min(xx), max(xx)]);
set(gca, 'XScale', 'log');

xlabel('number of test functions per marginal');
ylabel('objective');

% zoom-in of the first figure
zoomin = 5:9;

figure('Position', [400, 100, 400, 300]);
ha = tight_subplot(1, 1, [0, 0], [0.135, 0.020], [0.11, 0.025]);
axes(ha(1));

hold on;

handle_LB = plot(xx(zoomin), MMOT_LB_list(zoomin), ...
    'Marker', 'o', 'Color', 'blue', ...
    'LineStyle', '--', 'LineWidth', line_width);
handle_UB = errorbar(xx(zoomin), MMOT_UB_mean_list(zoomin), ...
    MMOT_UB_err(zoomin, 1), MMOT_UB_err(zoomin, 2), ...
    'Marker', 'none', 'Color', 'red', 'LineStyle', ':', ...
    'LineWidth', line_width);
handle_W2OTUB = errorbar(xx(zoomin), MMOT_W2OTUB_mean_list(zoomin), ...
    MMOT_W2OTUB_err(zoomin, 1), MMOT_W2OTUB_err(zoomin, 2), ...
    'Marker', 'none', 'Color', 'green', 'LineStyle', '-.', ...
    'LineWidth', line_width);

box on;
grid on;

legend([handle_UB, handle_W2OTUB, handle_LB], ...
    {'$\alpha^{\mathsf{UB}}$', ...
    '$\beta^{\mathsf{UB}}$', ...
    '$\alpha^{\mathsf{LB}}$'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 13);

set(gca, 'XTick', xx(zoomin));
xtickangle(25);
set(gca, 'XLim', [min(xx(zoomin)), max(xx(zoomin))]);
set(gca, 'XScale', 'log');

xlabel('number of test functions per marginal');
ylabel('objective');


% figure of sub-optimalities and error bounds

figure('Position', [800, 100, 400, 300]);
ha = tight_subplot(1, 1, [0, 0], [0.135, 0.020], [0.095, 0.025]);
axes(ha(1));

hold on;

handle_sub = errorbar(xx, MMOT_UB_mean_list - MMOT_LB_list, MMOT_UB_err(:, 1), MMOT_UB_err(:, 2), ...
    'Marker', 'none', 'Color', 'red', 'LineStyle', ':', 'LineWidth', line_width);
handle_W2OTsub = errorbar(xx, MMOT_W2OTUB_mean_list - MMOT_LB_list, MMOT_W2OTUB_err(:, 1), MMOT_W2OTUB_err(:, 2), ...
    'Marker', 'none', 'Color', 'green', 'LineStyle', '-.', 'LineWidth', line_width);
handle_th = plot(xx, MMOT_THEB_list, ...
    'Marker', 'diamond', 'Color', 'black', 'LineStyle', '-', 'LineWidth', line_width);

box on;
grid on;

legend([handle_sub, handle_W2OTsub, handle_th], ...
    {'$\tilde{\epsilon}_{\mathsf{sub}}$', ...
    '$\tilde{\xi}_{\mathsf{sub}}$', ...
    '$\epsilon_{\mathsf{theo}}$'}, ...
    'Location', 'northeast', ...
    'Interpreter', 'latex', ...
    'FontSize', 13);

set(gca, 'XTick', xx);
xtickangle(25);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');
set(gca, 'XLim', [min(xx), max(xx)]);
set(gca, 'YLim', [5e-5, 1e2]);

xlabel('number of test functions per marginal');
ylabel('sub-optimality');
