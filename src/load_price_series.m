function [mu, Sigma, names] = load_price_series(folder)
%LOAD_PRICE_SERIES Build return statistics from one price series per file.
%   [mu, Sigma, names] = LOAD_PRICE_SERIES(folder) reads every *.txt file in
%   folder (one closing price per line, all of equal length), computes simple
%   daily returns and returns their mean vector, covariance matrix and the
%   asset names (file names without extension), in alphabetical order.

files = dir(fullfile(folder, '*.txt'));
names = erase({files.name}, '.txt');

prices = zeros(0, numel(files));
for k = 1:numel(files)
    prices(:, k) = readmatrix(fullfile(folder, files(k).name)); %#ok<AGROW>
end

returns = diff(prices) ./ prices(1:end-1, :);
mu    = mean(returns)';
Sigma = cov(returns);
end
