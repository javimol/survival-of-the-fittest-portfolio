function [variance, w] = min_variance_weights(subset, mu, Sigma, target)
%MIN_VARIANCE_WEIGHTS Markowitz minimum-variance portfolio on a subset of assets.
%   [variance, w] = MIN_VARIANCE_WEIGHTS(subset, mu, Sigma, target) solves
%
%       min  w' * Sigma(subset,subset) * w
%       s.t. mu(subset)' * w = target
%            sum(w) = 1,  0 <= w <= 1
%
%   and returns the optimal variance and the weights of the assets in subset
%   (same order as subset). If the problem is infeasible, variance is Inf and
%   w is empty. This is the fitness function of the genetic algorithm.

mu = mu(:);
k  = numel(subset);

H   = 2 * Sigma(subset, subset);
Aeq = [mu(subset)'; ones(1, k)];
beq = [target; 1];

opts = optimoptions('quadprog', 'Display', 'off');
[w, variance, exitflag] = quadprog(H, zeros(k, 1), [], [], Aeq, beq, ...
                                   zeros(k, 1), ones(k, 1), [], opts);
if exitflag ~= 1
    variance = Inf;
    w = [];
end
end
