%RUN_ORLIB Cardinality-constrained frontiers for OR-Library port1..port4.
%   For every problem and every cardinality in K, sweeps the target return
%   with the genetic algorithm, writes each frontier to
%   results/portN_frontier_kK.csv ([return, variance] rows) and plots them.
%
%   This takes a long time (hours): every point of every frontier is one GA
%   run, and every GA fitness evaluation is one QP. Reduce K, PROBLEMS or
%   increase STEP for a quick look.

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root, 'src'));

PROBLEMS = 1:4;
K        = 3:9;
STEP     = 1e-4;

outDir = fullfile(root, 'results');
if ~exist(outDir, 'dir'), mkdir(outDir); end

figure
for p = PROBLEMS
    name = sprintf('port%d', p);
    [mu, Sigma] = load_orlib_portfolio(fullfile(root, 'data', 'orlib', [name '.txt']));

    frontiers = cell(1, numel(K));
    for i = 1:numel(K)
        fprintf('%s  k = %d ...\n', name, K(i));
        frontiers{i} = efficient_frontier(mu, Sigma, K(i), @solve_ga, STEP);
        writematrix(frontiers{i}, fullfile(outDir, sprintf('%s_frontier_k%d.csv', name, K(i))));
    end

    subplot(2, 2, p)
    plot_frontiers(frontiers, compose('k = %d', K), ...
                   sprintf('%s (N = %d)', name, numel(mu)));
end
