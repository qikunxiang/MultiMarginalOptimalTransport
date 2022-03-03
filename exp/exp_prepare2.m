% Preparation step 2: generating the necessary inputs for approximating the
% inverse cumulative distribution function of truncated mixtures of normal
% distributions to very high precision

load('exp/inputs.mat');

mixnorm_invcdf_cell = cell(N, 2);
cdf_granularity = 5;

for i = 1:N
    mu = mixnorm_mu_cell{i};
    sig = sqrt(mixnorm_sig2_cell{i});
    w = mixnorm_w_cell{i};

    cdf_bound1 = normcdf((mixnorm_trunc(1) - mu) ./ sig);
    cdf_bound2 = normcdf((mixnorm_trunc(2) - mu) ./ sig);
    normconst = cdf_bound2 - cdf_bound1;

    t_func = @(v)(sum(((normcdf(((v' - mu) ./ sig)) - cdf_bound1) ...
        ./ normconst) .* w, 1)');

    xx = (0:10^cdf_granularity)' / 10^cdf_granularity ...
        * (mixnorm_trunc(2) - mixnorm_trunc(1)) + mixnorm_trunc(1);
    yy = t_func(xx);

    zero_id = find(yy == 0, 1, 'last');
    yy = yy(zero_id:end);
    xx = xx(zero_id:end);

    [yy, ia] = unique(yy);
    xx = xx(ia);

    mixnorm_invcdf_cell{i, 1} = yy;
    mixnorm_invcdf_cell{i, 2} = xx;
end

save('exp/invcdf.mat', 'mixnorm_invcdf_cell', '-v7.3');

