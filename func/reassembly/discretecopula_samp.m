function [samp, samp_atomid] = discretecopula_samp(atoms, prob, num)
% Generate random samples from the copula of a given discrete distribution
% Inputs: 
%       atoms: atoms in the discrete joint distribution, each row
%       represents an atom
%       prob: the corresponding probabilities at the atoms
%       num: number of samples
% Outputs: 
%       samp: samples, each row represents a sample
%       samp_atomid: samples of the associated atom indices


[atomno, d] = size(atoms);

[~, temp] = sort(atoms);
probs_marg = prob(temp);
[~, atoms_order] = sort(temp);
atoms_order_lin = atoms_order + (0:d - 1) * atomno;
probs_cum = cumsum(probs_marg, 1);
atoms_unit = probs_cum(atoms_order_lin);
probs_rem = probs_marg(atoms_order_lin);

samp_atomid = randsample(atomno, num, true, prob);
samp = atoms_unit(samp_atomid, :) - rand(num, d) ...
    .* probs_rem(samp_atomid, :);

end

