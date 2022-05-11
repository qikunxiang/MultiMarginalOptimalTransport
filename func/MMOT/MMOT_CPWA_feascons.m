function [A, b] = MMOT_CPWA_feascons(x, knots, CPWA)
% Generate linear feasibility constraints in the LSIP problem for MMOT with
% CPWA cost function
% Inputs: 
%       x: each row is a point where a constraint should be generated
%       knots: the knots in each dimension
%       CPWA: the specification of the CPWA payoff function
% Outputs: 
%       A, b: the resulting linear constraints A * x <= b

n = size(x, 1);
N = length(knots);

b = CPWA_nonsep_eval(x, CPWA);

% sanitize the output to prevent potential numerical issues
b(abs(b) < 1e-9) = 0;

A_cell = cell(N, 1);

for i = 1:N
    basis = momentset1d_basisfunc_bounded(x(:, i), knots{i});

    if i == 1
        % include the first knot for the first dimension
        basis = [1 - sum(basis, 2), basis]; %#ok<AGROW> 
    end

    % sanitize the output to prevent potential numerical issues
    basis(abs(basis) < 1e-9) = 0;
    A_cell{i} = sparse(basis);
end

A = horzcat(A_cell{:});

end

