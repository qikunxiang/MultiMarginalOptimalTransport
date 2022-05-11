function val_rep = MMOT_CPWA_primal_approx(CPWA, atoms, probs, ...
    marginv, sampno, rep_no, batchsize, rand_stream, samp_mode, ...
    run_in_parallel)
% Compute an upper bound on the MMOT problem with CPWA cost function by
% Monte Carlo sampling from a primally feasible measure
% Inputs: 
%       CPWA: the specification of the CPWA cost function
%       atoms: the atoms of the unmodified measure
%       probs: the corresponding probabilities of the unmodified measure
%       marginv: the inverse cdf of the marginal distributions represented
%       using a single function handle (i, u) -> marginv(i, u)
%       sampno: the number of samples used in the Monte Carlo integration
%       rep_no: the number of repetitions to approximate the distribution
%       of the Monte Carlo integration
%       batchsize: the maximum number of samples generated in each batch,
%       used to prevent memory surges (default is 1e6)
%       rand_stream: RandStream object to guarantee reproducibility
%       samp_mode: the copula used for generating uniform random variables,
%       either 'comonotone' or 'independent' (default is 'independent')
%       run_in_parallel: a boolean parameter indicating whether to compute
%       the repetitions in parallel (default is false)
% Outputs: 
%       val_rep: a list of expected values from repeated trials of Monte
%       Carlo integration

if ~exist('batchsize', 'var') || isempty(batchsize)
    batchsize = 1e6;
end

if ~exist('samp_mode', 'var') || isempty(samp_mode)
    samp_mode = 'comonotone';
end

if ~exist('run_in_parallel', 'var') || isempty(run_in_parallel)
    run_in_parallel = false;
end

N = size(atoms, 2);

val_rep = zeros(rep_no, 1);

if run_in_parallel

    rs = parallel.pool.Constant(rand_stream);

    parfor_progress(rep_no);
    parfor rep_id = 1:rep_no
        substream = rs.Value;
        substream.Substream = rep_id;
        
        val_sum = 0;
        sampno_remain = sampno;
        
        % cap the number of samples generated at once at the batch size
        while sampno_remain > 0
            sampno_batch = min(sampno_remain, batchsize);
            copula_samples = discretecopula_samp(atoms, probs, ...
                sampno_batch, substream, samp_mode);
            copula_samples = max(copula_samples, 0);
            copula_samples(copula_samples >= 1) = 1;

            samples = zeros(sampno_batch, N);

            for i = 1:N
                samples(:, i) = marginv(i, copula_samples(:, i)); ...
                    %#ok<PFBNS>
            end

            val_samples = CPWA_nonsep_eval(samples, CPWA);
            
            % accumulate the samples
            val_sum = val_sum + sum(val_samples);
            
            % subtract the number of samples generated
            sampno_remain = sampno_remain - sampno_batch;
        end
        
        val_rep(rep_id) = val_sum / sampno;
        parfor_progress;
    end
    parfor_progress(0);
else
    parfor_progress(rep_no);
    for rep_id = 1:rep_no
        rand_stream.Substream = rep_id;
        
        val_sum = 0;
        sampno_remain = sampno;
        
        % cap the number of samples generated at once at the batch size
        while sampno_remain > 0
            sampno_batch = min(sampno_remain, batchsize);
            copula_samples = discretecopula_samp(atoms, probs, ...
                sampno_batch, rand_stream, samp_mode);
            copula_samples = max(copula_samples, 0);
            copula_samples(copula_samples >= 1) = 1;

            samples = zeros(sampno_batch, N);

            for i = 1:N
                samples(:, i) = marginv(i, copula_samples(:, i));
            end

            val_samples = CPWA_nonsep_eval(samples, CPWA);
            
            % accumulate the samples
            val_sum = val_sum + sum(val_samples);
            
            % subtract the number of samples generated
            sampno_remain = sampno_remain - sampno_batch;
        end
        
        val_rep(rep_id) = val_sum / sampno;

        parfor_progress;
    end

    parfor_progress(0);
end

end

