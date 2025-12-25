% Plot the running time of the cutting-plane algorithm against the number of time steps

CONFIG = IFMM_config();

load(CONFIG.SAVEPATH_INPUTS);
load(CONFIG.SAVEPATH_OUTPUTS);

running_time = zeros(length(timestep_num_list), 2);

for arr_id = 1:length(arrangements)
    for ts_id = 1:length(timestep_num_list)
        for tf_id = 1:length(testfunc_knot_num_list)
            running_time(ts_id, arr_id) = running_time(ts_id, arr_id) ...
                + output_cell{arr_id}{ts_id, tf_id}.total_time;
        end
    end
end

figure('Position', [400, 100, 400, 400]);
ha = tight_subplot(1, 1, [0, 0], [0.08, 0.035], [0.095, 0.025]);
axes(ha(1));

hold on;
plot(timestep_num_list, running_time(:, 1));
plot(timestep_num_list, running_time(:, 2));
set(gca, 'XScale', 'log');
set(gca, 'YScale', 'log');