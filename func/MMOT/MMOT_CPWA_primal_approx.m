function val_rep = MMOT_CPWA_primal_approx(CPWA, atoms, probs, ...
    marginv, sampno, repno, batchsize, parallel)
% Compute an upper bound on the MMOT problem with CPWA cost function by
% Monte Carlo sampling from a primally feasible measure
% Inputs: 
%       CPWA: the specification of the CPWA cost function
%       atoms: the atoms of the unmodified measure
%       probs: the corresponding probabilities of the unmodified measure
%       marginv: the inverse cdf of the marginal distributions represented
%       using a single function handle (i, u) -> marginv(i, u)
%       sampno: the number of samples used in the Monte Carlo integration
%       repno: the number of repetitions to approximate the distribution of
%       the Monte Carlo integration
%       batchsize: the maximum number of samples generated in each batch,
%       used to prevent memory surges (default is 1e6)
%       parallel: a boolean parameter indicating whether to compute the
%       repetitions in parallel (default is false)
% Outputs: 
%       val_rep: a list of expected values from repeated trials of Monte
%       Carlo integration

if ~exist('batchsize', 'var') || isempty(batchsize)
    batchsize = 1e6;
end

if ~exist('parallel', 'var') || isempty(parallel)
    parallel = false;
end

N = size(atoms, 2);

% check the inputs
% assert(length(marginv) == N, 'marginal distributions mis-specified');

val_rep = zeros(repno, 1);

if parallel
    parfor_progress(repno);
    parfor repid = 1:repno
        stream = RandStream.getGlobalStream();
        stream.Substream = repid;
        
        val_sum = 0;
        sampno_remain = sampno;
        
        % cap the number of samples generated at once at the batch size
        while sampno_remain > 0
            sampno_batch = min(sampno_remain, batchsize);
            copula_samples = discretecopula_samp(atoms, probs, ...
                sampno_batch);
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
        
        val_rep(repid) = val_sum / sampno;
        parfor_progress;
    end
    parfor_progress(0);
else
    parfor_progress(repno);
    for repid = 1:repno
        stream = RandStream.getGlobalStream();
        stream.Substream = repid;
        
        val_sum = 0;
        sampno_remain = sampno;
        
        % cap the number of samples generated at once at the batch size
        while sampno_remain > 0
            sampno_batch = min(sampno_remain, batchsize);
            copula_samples = discretecopula_samp(atoms, probs, ...
                sampno_batch);
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
        
        val_rep(repid) = val_sum / sampno;

        parfor_progress;
    end

    parfor_progress(0);
end

end

