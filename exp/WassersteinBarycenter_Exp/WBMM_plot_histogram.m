% Plot the histogram of computed approximate Wasserstein barycenter

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);
load(CONFIG.SAVEPATH_UB);
load(CONFIG.SAVEPATH_W2OTUB);

x_axis_lim = [-0.0, 3.0];
y_axis_lim = [-0.0, 3.0];

x_label_cell = cell(test_num, 1);

for test_id = 1:test_num
    marg_testfuncs_num = size(marg_testfuncs_cell{test_id}{1}{1}, 1) - 1;

    x_label_cell{test_id} = sprintf('$m_0=%d$', marg_testfuncs_num);
end

dens_max = max(max([WB_histpdf_cell{end - 1}; WB_W2OT_histpdf_cell{end - 1}]));

figure('Position', [100, 700, 1000, 450]);
ha = tight_subplot(2, 5, [0.09, 0.015], [0.07, 0.010], [0.012, 0.003]);
ha(10).Position(1) = 2;


for test_id = 1:test_num
    axes(ha(test_id));
    
    if test_id > 5
        ha(test_id).Position(1) = ha(test_id).Position(1) + 0.090;
    end

    hold on;

    plot_color = pcolor(WB_plot_hist_grid_x, WB_plot_hist_grid_y, WB_histpdf_cell{test_id}');
    plot_color.EdgeColor = 'interp';
    plot_color.FaceColor = 'interp';

    box on;
    colormap('hot');

    clim([0, dens_max]);
    set(gca, 'XTick', 0:3);
    set(gca, 'YTick', 0:3);
    set(gca, 'XLim', x_axis_lim);
    set(gca, 'YLim', y_axis_lim);

    xlabel(x_label_cell{test_id}, 'Interpreter', 'latex');
end

cb = colorbar(ha(test_id), 'manual');
cb.Position = [0.897, 0.068, 0.015, 0.418];



figure('Position', [100, 100, 1000, 450]);
ha = tight_subplot(2, 5, [0.09, 0.015], ...
    [0.07, 0.010], [0.012, 0.003]);
ha(10).Position(1) = 2;


for test_id = 1:test_num
    axes(ha(test_id));
    hold on;

    if test_id > 5
        ha(test_id).Position(1) = ha(test_id).Position(1) + 0.090;
    end

    plot_color = pcolor(WB_plot_hist_grid_x, ...
        WB_plot_hist_grid_y, ...
        WB_W2OT_histpdf_cell{test_id}');
    plot_color.EdgeColor = 'interp';
    plot_color.FaceColor = 'interp';

    box on;
    colormap('hot');

    clim([0, dens_max]);
    set(gca, 'XTick', 0:3);
    set(gca, 'YTick', 0:3);
    set(gca, 'XLim', x_axis_lim);
    set(gca, 'YLim', y_axis_lim);

    xlabel(x_label_cell{test_id}, 'Interpreter', 'latex');
end

cb = colorbar(ha(test_id), 'manual');
cb.Position = [0.897, 0.068, 0.015, 0.418];