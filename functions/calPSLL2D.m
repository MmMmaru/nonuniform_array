function psll = calPSLL2D(points, thetaScanDeg, phiScanDeg, f0, Lx, Ly)
    % 输入为m, f = Hz

    c0 = 3*1e8;
    lambda0 = c0/(f0);
    k0 = 2*pi/lambda0;
    
    % 建立坐标系
    % 创建网格
    
    hyperuniform_points_normed = points .* [Lx, Ly] - [Lx/2, Ly/2];
    
    % 选择点
    points = hyperuniform_points_normed;
    N = size(points, 1);
    
    % 生成uv
    resolution = 1;
    Theta = linspace(-90, 90, 180 * resolution + 1);Phi = linspace(0, 179, 179 * resolution + 1);
    [theta, phi] = meshgrid(Theta, Phi);
    
    % 扫描角
    u = sind(theta) .* cosd(phi);
    v = sind(theta) .* sind(phi);
    F = zeros(size(theta, 1),size(theta, 2));
    u0 = [sind(thetaScanDeg)*cosd(phiScanDeg), sind(thetaScanDeg)*sind(phiScanDeg), 0];
    for n = 1:N
        rn = [points(n,:),0];
        An = 1 ;
        phase_n = k0 * (-dot(rn, u0));
        F = F + An .* exp(1j * k0 * (rn(1) * u + rn(2) * v) + 1j * phase_n);
    end
    S = abs(F);
    P_rad = sqrt(N);
    intensity_pattern = S/P_rad; 
    normalized_intensity_pattern = intensity_pattern; % 归一化处理
    dB_intensity_pattern = 20 * log10(normalized_intensity_pattern);
    peaks = findpeaks(dB_intensity_pattern((phiScanDeg) * resolution +1, :),'SortStr','descend');
    psll = peaks(2)-peaks(1);


    plot(Theta, dB_intensity_pattern((phiScanDeg) * resolution +1, :), 'DisplayName', "theta=" + num2str(thetaScanDeg));
    title("phi=" + num2str(phiScanDeg) +" radiation pattern");
    xlim([-90, 90]);
    ylim([-20, 20]);
end