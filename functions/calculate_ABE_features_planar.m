function binary_grid = calculate_ABE_features_planar(points, L, interval)
    % 统一单位mm
    % L:考虑区域的边长
    % interval:间距
    for i = 1:size(points, 1)
        all_coords = points;
        abe_coord = all_coords(i, :);
        other_coords = all_coords;
        other_coords(i, :) = []; % 移除自身
        n_grid = round(L/interval)+1;
    
        x_edges = linspace(-L/2, L/2, n_grid);
        y_edges = linspace(-L/2, L/2, n_grid);
        
        % 计算相对坐标 (极坐标)
        relative_coords_cart = other_coords - abe_coord;
        counts_grid = histcounts2(relative_coords_cart(:,1), relative_coords_cart(:,2), x_edges, y_edges);
        
        binary_grid(i,:,:) = flipud(double(counts_grid > 0)'); 
    end
end

% --- 使用示例 ---
% 假设 N 是单元数, K, M, Rc 已定义
% all_ln = zeros(2 + K*M, N); % 预分配内存
% for n = 1:N
%     all_ln(:, n) = calculate_ABE_features(n, array_geometry_sample_i, K, M, Rc);
% end
