function plot_inUV(g_mag_linear, angles_uv)
    
    % 可选：归一化到最大值为 0 dB (如果需要相对方向图)
    % g_ave_mag_db = g_ave_mag_db - max(g_ave_mag_db);
    u = angles_uv(:,1);
    v = angles_uv(:,2);
    % 1. 定义规则的 uv 网格
    nGridPoints = 101; % 网格精细度，例如 201x201
    u_grid_vec = linspace(-1, 1, nGridPoints);
    v_grid_vec = linspace(-1, 1, nGridPoints);
    [U_grid, V_grid] = meshgrid(u_grid_vec, v_grid_vec);
    
    % 2. 使用 griddata 进行插值
    fprintf('正在将数据插值到 uv 网格...\n');
    % F = scatteredInterpolant(u, v, g_ave_mag_db, 'natural'); % 备选插值方法
    % G_ave_mag_db_grid = F(U_grid, V_grid);
    G_ave_mag_db_grid = griddata(u, v, g_mag_linear, U_grid, V_grid, 'natural'); % 'natural' 是常用的插值方法
    
    % 3. 将可见区外 (u^2+v^2 > 1) 的值设为 NaN，使其不显示
    outside_visible = (U_grid.^2 + V_grid.^2) > 1;
    G_ave_mag_db_grid(outside_visible) = NaN;
    
    % 4. 绘图 (使用 pcolor)
    figure;
    pcolor(U_grid, V_grid, G_ave_mag_db_grid);
    shading interp; % 平滑颜色过渡，隐藏网格线
    axis equal;
    axis([-1 1 -1 1]);
    hold on;
    
    % 添加颜色条和颜色映射
    hcb = colorbar;
    ylabel(hcb, 'Magnitude');
    colormap('jet');
%     clim([-20, 20]);
    hold off;
    fprintf('已生成 uv 域插值 pcolor 图。\n');
end