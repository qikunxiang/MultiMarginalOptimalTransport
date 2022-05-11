load('exp/exp_inputs.mat', 'knot_no_list', 'err_bound_list');
load('exp/exp_rst_LB.mat', 'LSIP_LB_list');
load('exp/exp_rst_UB.mat', 'MMOT_UB_errbar', 'MMOT_UB_samps_cell');

xx = knot_no_list;

figure('Position', [100, 100, 400, 400]);
tight_subplot(1, 1, [0, 0], [0.09, 0.020], [0.11, 0.025]);
hold on;
handle_LB = plot(xx, LSIP_LB_list, '-*', 'Color', 'blue');
UB_mean = zeros(length(knot_no_list), 1);

for stepid = 1:length(knot_no_list)
    UB_mean(stepid) = mean(MMOT_UB_samps_cell{stepid});
end

UB_err_neg = UB_mean - MMOT_UB_errbar(:, 1);
UB_err_pos = MMOT_UB_errbar(:, 2) - UB_mean;
handle_UB = errorbar(xx, UB_mean, UB_err_neg, UB_err_pos, ...
    'Marker', 'none', 'LineStyle', ':', 'Color', 'red');
set(gca, 'XScale', 'log');
set(gca, 'XLim', [xx(1), xx(end)]);

legend([handle_UB, handle_LB], 'upper bound', 'lower bound');

xlabel('number of knots');
ylabel('cost');



figure('Position', [500, 100, 400, 400]);
tight_subplot(1, 1, [0, 0], [0.09, 0.020], [0.10, 0.025]);
hold on;

gap_mean = UB_mean - LSIP_LB_list;
gap_err_neg = UB_err_pos;
gap_err_pos = UB_err_neg;

handle_theory = plot(xx, err_bound_list, '--^', 'Color', 'magenta');
handle_actual = errorbar(xx, gap_mean, gap_err_neg, gap_err_pos, ...
    'Marker', 'none', 'LineStyle', '-', 'Color', 'black');
set(gca, 'XLim', [xx(1), xx(end)]);
set(gca, 'YLim', [min(gap_mean - gap_err_neg) * 0.5, ...
    max(err_bound_list) * 2]);
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');

legend([handle_theory, handle_actual], 'theoretical upper bound', ...
    'actual difference');

xlabel('number of knots');
ylabel('difference between the bounds');