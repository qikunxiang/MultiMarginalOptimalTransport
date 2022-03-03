function [atoms_new, probs_new] = reassembly_increment(knots, ...
    knots_new, expval_new, atoms, probs)
% Compute the reassembly of a discrete measure with new discrete marginals
% in order to construct a new primal feasible solution with the given
% discrete marginals
% Inputs: 
%       knots: the old knots
%       knots_new: the new knots (must be supersets of the old knots)
%       expval_new: the expected values of the new CPWA basis functions
%       atoms: the atoms in the primal feasible solution (a measure) before
%       the update
%       probs: the corresponding probabilities in the primal feasible
%       solution (a measure) before the update
% Outputs: 
%       atoms_new: the atoms in the updated primal feasible solution
%       probs_new: the atoms in the updated primal feasible solution

N = length(knots);

% check the inputs
assert(length(knots_new) == N, 'new knots mis-specified');
assert(size(atoms, 2) == N, 'atoms mis-specified');
assert(size(atoms, 1) == length(probs), 'probabilities mis-specified');

% normalize the probabilities (in case there is a small numerical error)
probs = probs / sum(probs);

coef_num_list_old = zeros(N, 1);
coef_num_list_new = zeros(N, 1);

for i = 1:N
    assert(all(ismember(knots{i}, knots_new{i})), ...
        'new knots are not supersets of old knots');
    assert(abs(knots{i}(end) - knots_new{i}(end)) <= eps, ...
        'the right-most knot must be the same');
    coef_num_list_old(i) = length(knots{i});
    coef_num_list_new(i) = length(knots_new{i});
    assert(length(expval_new{i}) == coef_num_list_new(i), ...
        'new expected values mis-specified');
end

% compute the updated measure

% remove atoms with zero probability
nonzero_list = probs > 0;
probs_new = probs(nonzero_list);
atoms_new = atoms(nonzero_list, :);

% begin the loop to compute a reassembly
for i = 1:N
    % compute a marginal distribution in the moment set
    [marg_atoms_i, marg_prob_i] = momentset1d_measinit_bounded( ...
        knots_new{i}, expval_new{i});
    
    % sort the atoms into ascending order in the i-th dimension
    [~, asc_order] = sort(atoms_new(:, i), 'ascend');
    atoms_sorted = atoms_new(asc_order, :);
    prob_sorted = probs_new(asc_order);
    
    % the atoms and probabilities in the glued distribution
    atomno_max = size(atoms_sorted, 1) * length(marg_atoms_i);
    atoms_glue = zeros(atomno_max, N + 1);
    prob_glue = zeros(atomno_max, 1);
    
    % initialize the counters
    counter_joint = 1;
    counter_marg = 1;
    counter_glue = 0;
    
    % begin the loop to couple the marginals
    while true
        % take out one atom
        prob_min = min(prob_sorted(counter_joint), ...
            marg_prob_i(counter_marg));
        
        % record the atom and its probability
        counter_glue = counter_glue + 1;
        atoms_glue(counter_glue, :) = [atoms_sorted(counter_joint, :), ...
            marg_atoms_i(counter_marg)];
        prob_glue(counter_glue) = prob_min;
        
        % decrease the probability from the remaining probabilities
        prob_sorted(counter_joint) = prob_sorted(counter_joint) ...
            - prob_min;
        marg_prob_i(counter_marg) = marg_prob_i(counter_marg) ...
            - prob_min;
        
        % advance the counters
        if prob_sorted(counter_joint) == 0
            counter_joint = counter_joint + 1;
        end
        
        if marg_prob_i(counter_marg) == 0
            counter_marg = counter_marg + 1;
        end
        
        if counter_joint > size(atoms_sorted, 1) ...
                || counter_marg > length(marg_atoms_i)
            break;
        end
    end
    
    % update the distribution
    atoms_glue = atoms_glue(1:counter_glue, :);
    atoms_new = atoms_glue(:, 1:N);
    atoms_new(:, i) = atoms_glue(:, end);
    probs_new = prob_glue(1:counter_glue);
end

end

