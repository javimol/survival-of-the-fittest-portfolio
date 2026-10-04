function [w, variance] = solve_exhaustive(mu, Sigma, k, target)
%SOLVE_EXHAUSTIVE Exact cardinality-constrained portfolio by full enumeration.
%   [w, variance] = SOLVE_EXHAUSTIVE(mu, Sigma, k, target) solves the
%   minimum-variance QP for every one of the nchoosek(N, k) subsets of k
%   assets and keeps the best one. w is the full N-by-1 weight vector.
%   variance is Inf if no subset can reach the target return.
%
%   Only practical for small N (e.g. the 5 Madrid stocks); use SOLVE_GA for
%   the OR-Library problems.

n = numel(mu);
subsets  = nchoosek(1:n, k);
w        = zeros(n, 1);
variance = Inf;

for i = 1:size(subsets, 1)
    [v, ws] = min_variance_weights(subsets(i, :), mu, Sigma, target);
    if v < variance
        variance = v;
        w = zeros(n, 1);
        w(subsets(i, :)) = ws;
    end
end
end
