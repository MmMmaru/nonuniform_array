function ln = calculate_xy_features(element_index, all_coords, N, large_element_coords)
    % 统一单位mm
    % element_index: 当前ABE的索引
    % all_coords: 阵列中所有单元的坐标 [N x 2]
    % K: 分区
    % M: 分区向量长度
    % Rc: 耦合区域半径
    % 返回相对坐标

    abe_coord = all_coords(element_index, :);
    other_coords = all_coords;
    other_coords(element_index, :) = []; % 移除自身
    N_total = size(all_coords, 1);

    % 计算相对坐标 (极坐标)
    relative_coords_cart = other_coords - abe_coord;
    [phi, r] = cart2pol(relative_coords_cart(:,1), relative_coords_cart(:,2));

    % 初始化相对距离特征向量
    r_relative_inv = zeros(N, 1);
    [sorted_r, sort_idx] = sort(r);
    coord1 = relative_coords_cart(sort_idx(1:N),:);

    % 计算相对坐标 (极坐标)
%     relative_coords_cart = large_element_coords - abe_coord;
%     [phi, r] = cart2pol(relative_coords_cart(:,1), relative_coords_cart(:,2));
%     phi = mod(phi, 2*pi); % 确保角度在 [0, 2*pi)
%     r_large = zeros(N, 1);
%     [sorted_r, sort_idx] = sort(r);
%     coord2 = relative_coords_cart(sort_idx(1:N), :);
    % 组合特征: 绝对位置 + 相对位置倒数 + 大尺寸单元位置
    ln = coord1.'; % 列向量形式
    % ln = ln(:);
    % N*2
end