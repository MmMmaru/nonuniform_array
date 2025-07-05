function [PSLL, Gain] = calPSLL3D(points, thetaScanDeg, phiScanDeg, f0, Lx, Ly, Gn, angles_uv)
% 物理量
% 输入points 单位为mm

    c0 = 3*1e8;
    lambda0 = c0/(f0);
    k0 = 2*pi/lambda0;
    
    % 建立坐标系
    % 创建网格
    
    hyperuniform_points_normed = points .* [Lx, Ly];
    
    % 选择点
    points = hyperuniform_points_normed;
    N = size(points, 1);
    
    % 生成uv

    % u = angles_uv(:,1);
    % v = angles_uv(:,2);
    grid_points = 64;
    [U_grid,V_grid] = meshgrid(linspace(-1,1,grid_points), linspace(1,-1,grid_points));
    % 扫描角
    F = zeros(size(U_grid, 1),size(U_grid, 2));
    u0 = [sind(thetaScanDeg)*cosd(phiScanDeg), sind(thetaScanDeg)*sind(phiScanDeg), 0];
    warning('off');
    for n = 1:N
        rn = [points(n,:),0];

%         Gn_ = scatteredInterpolant(u, v, G_mag_all(:, n), 'natural'); % 备选插值方法
%         Gn = Gn_(U_grid, V_grid);
        % Gn = griddata(u, v, G_mag_all(:, n), U_grid, V_grid, 'natural');
        An = Gn ;
        phase_n = k0 * (-dot(rn, u0));
        F = F + An .* exp(1j * k0 * (rn(1) * U_grid + rn(2) * V_grid) + 1j * phase_n);
    end
    U = abs(F) .^2 /2 /377;
    Gain = 4*pi*U/ N; 
    Gain = 10 * log10(Gain);
    Gain(isnan(Gain)) = -20;
    
%     4. 绘图 (使用 pcolor)
    figure;
    pcolor(U_grid, V_grid, Gain);
    shading interp; % 平滑颜色过渡，隐藏网格线
    axis equal;
    axis([-1 1 -1 1]);
    hold on;
    
    % 添加颜色条和颜色映射
    hcb = colorbar;
    ylabel(hcb, 'Magnitude');
    colormap('jet');
    clim([-20, 20]);

    mask =  imregionalmax(Gain);
    peaks = Gain(mask);
    sorted_peaks = sort(peaks, 'descend');
    PSLL = sorted_peaks(2) - sorted_peaks(1);

    fprintf("maxGain = %f", sorted_peaks(1));
end