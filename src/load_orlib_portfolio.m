function [mu, Sigma] = load_orlib_portfolio(filename)
%LOAD_ORLIB_PORTFOLIO Read a Beasley OR-Library portfolio file (port1..port5).
%   [mu, Sigma] = LOAD_ORLIB_PORTFOLIO(filename) returns the column vector of
%   mean weekly returns and the covariance matrix.
%
%   File format:
%     N
%     mean_i  std_i      (one line per asset)
%     i  j  rho_ij       (one line per pair, i <= j)

fid = fopen(filename, 'r');
if fid < 0
    error('load_orlib_portfolio:io', 'Cannot open %s', filename);
end
closer = onCleanup(@() fclose(fid));

n     = fscanf(fid, '%d', 1);
stats = fscanf(fid, '%f', [2 n])';
pairs = fscanf(fid, '%f', [3 Inf])';

mu = stats(:, 1);
sd = stats(:, 2);

% Upper triangle (with diagonal) from the file, then mirror it.
rho = zeros(n);
rho(sub2ind([n n], pairs(:, 1), pairs(:, 2))) = pairs(:, 3);
rho = rho + triu(rho, 1)';

Sigma = rho .* (sd * sd');
end
