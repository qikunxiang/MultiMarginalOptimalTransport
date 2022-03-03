function v = CPWA_diff_eval(x, c, knots, knots_val, CPWA)
% Evaluate the difference between a separable CPWA function and a
% non-separable CPWA function
% Inputs: 
%       x: inputs, each row is an input, columns are the dimensions
%       knots: cell array containing the knots for the one-dimensional CPWA
%       functions in each dimension; the first and last knots also specify
%       the boundaries of the box
%       knots_val: cell array containing the values of the one-dimensional
%       CPWA functions in each dimension
%       CPWA: the specification of the non-separable CPWA function
% Outputs: 
%       v: the function values

[n, K] = size(x);
v = zeros(n, 1) + c;

for i = 1:K
    v = v + sum(min(max((x(:, i) - knots{i}(1:end - 1)') ...
        ./ diff(knots{i})', 0), 1) .* diff(knots_val{i})', 2) ...
        + knots_val{i}(1);
end

if isfield(CPWA, 'plus')
    for s = 1:size(CPWA.plus, 1)
        v = v - max(x * CPWA.plus{s, 1}' + CPWA.plus{s, 2}', [], 2);
    end
end

if isfield(CPWA, 'minus')
    for s = 1:size(CPWA.minus, 1)
        v = v + max(x * CPWA.minus{s, 1}' + CPWA.minus{s, 2}', [], 2);
    end
end
end

