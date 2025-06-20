function plotg2(points, dr, rmin, rmax)
    N = size(points,1);
    x = points(:,1);
    y = points(:,2);
    edges = rmin:dr:rmax; 
    rmax = edges(end);
    binCounts = zeros(1, length(edges)-1);
    for i = 1:size(points,1)-1
        for j = i+1:size(points,1)        
            dx = abs(x(i) - x(j));
            dy = abs(y(i) - y(j));
    
            % Apply periodic boundary
            dx = min(dx, 1 - dx);
            dy = min(dy, 1 - dy);
            d = sqrt(dx^2 + dy^2);
            if d < rmax
                bin = floor(abs(d-rmin)/dr) + 1;
                binCounts(bin) = binCounts(bin) + 2;
            end
        end
    end
    rho = 1;
    r = (edges(1:end-1) + edges(2:end))/2;
    normFactor = N;
    g2 = binCounts ./ normFactor;
    % 绘图
    figure;
    bar(6.5*r, g2,'LineWidth', 2);
    xlabel('r');
    ylabel('g_2(r)');
    title('Radial Distribution Function');
    grid on;
end

