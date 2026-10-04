function plot_frontiers(frontiers, labels, plotTitle)
%PLOT_FRONTIERS Plot several frontiers on the current axes.
%   frontiers is a cell array of [return, variance] matrices (as returned by
%   EFFICIENT_FRONTIER) and labels a cell array of legend entries.

hold on
for i = 1:numel(frontiers)
    f = frontiers{i};
    ok = isfinite(f(:, 2));
    plot(f(ok, 2), f(ok, 1), '.-', 'DisplayName', labels{i});
end
hold off
grid on
xlabel('Variance of return')
ylabel('Mean return')
title(plotTitle)
legend('Location', 'southeast')
end
