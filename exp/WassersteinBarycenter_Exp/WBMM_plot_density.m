% Plot the density functions of the marginals

CONFIG = WBMM_config();

load(CONFIG.SAVEPATH_INPUTS);

% instantiate the marginals
marg_cell = cell(marg_num, 1);
marg_dens_cell = cell(marg_num, 1);

dens_min = inf;
dens_max = 0;

for marg_id = 1:marg_num
    Meas = ProbMeas2DCPWADens(marg_vertices_cell{marg_id}, ...
        marg_triangles_cell{marg_id}, ...
        marg_density_cell{marg_id});
    marg_cell{marg_id} = Meas;

    marg_dens_cell{marg_id} = reshape( ...
        Meas.densityFunction(meas_plot_grid), ...
        meas_plot_y_num, meas_plot_x_num);

    dens_min = min(dens_min, min(min(marg_dens_cell{marg_id})));
    dens_max = max(dens_max, max(max(marg_dens_cell{marg_id})));
end

x_axis_lim = [-0.00, 3.00];
y_axis_lim = [-0.00, 3.00];


figure('Position', [100, 100, 1250, 250]);

ha = tight_subplot(1, 5, [0.0, 0.018], [0.135, 0.017], [0.012, 0.04]);

for marg_id = 1:marg_num
    axes(ha(marg_id));
    hold on;
    plot_color = pcolor(meas_plot_grid_x, meas_plot_grid_y, ...
        marg_dens_cell{marg_id});
    plot_color.EdgeColor = 'interp';
    plot_color.FaceColor = 'interp';
    clim([0, dens_max]);

    box on;
    colormap('hot');

    set(gca, 'XLim', x_axis_lim);
    set(gca, 'YLim', y_axis_lim);
    set(gca, 'XTick', 0:3);
    set(gca, 'YTick', 0:3);
    xlabel(sprintf('$\\mu_{%d}$', marg_id), ...
        'Interpreter', 'latex', 'FontSize', 14);
end

cb = colorbar(ha(end), 'manual');
cb.Position = [0.966, 0.134, 0.014, 0.850];