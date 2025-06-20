function ln = calculate_ABE_features_1(element_index, all_coords, K, M, Rc, lambda, large_element_coords, Lx, Ly)
    % 统一单位mm
    % element_index: 当前ABE的索引
    % all_coords: 阵列中所有单元的坐标 [N x 2]
    % K: 每个扇区考虑的最近邻居数
    % M: 耦合区域划分的扇区数
    % Rc: 耦合区域半径

    abe_coord = all_coords(element_index, :);
    other_coords = all_coords;
    other_coords(element_index, :) = []; % 移除自身
    N_total = size(all_coords, 1);

    % 计算相对坐标 (极坐标)
    relative_coords_cart = other_coords - abe_coord;
    [phi, r] = cart2pol(relative_coords_cart(:,1), relative_coords_cart(:,2));
    phi = mod(phi, 2*pi); % 确保角度在 [0, 2*pi)

    % 初始化相对距离特征向量
    r_relative_inv = zeros(M * K, 1);

    % 分扇区处理
    delta_phi = 2*pi / M;
    for m = 1:M
        phi_min = (m-1) * delta_phi;
        phi_max = m * delta_phi;

        % 找到属于该扇区且在 Rc 内的单元索引
        in_section_indices = find(phi >= phi_min & phi < phi_max & r <= Rc);

        if ~isempty(in_section_indices)
            % 获取这些单元的距离并排序
            r_in_section = r(in_section_indices);
            [sorted_r, sort_idx] = sort(r_in_section);

            % 取前 K 个
            num_found = length(sorted_r);
            num_to_take = min(num_found, K);

            start_idx_in_vec = (m-1)*K + 1;
            end_idx_in_vec = start_idx_in_vec + num_to_take - 1;
            
            sorted_r = sorted_r / lambda;
            r_relative_inv(start_idx_in_vec : end_idx_in_vec) = 1./sorted_r(1:num_to_take);
            % 其他默认为 0 (对应 1/inf)
        end
    end

    % 计算相对坐标 (极坐标)
    relative_coords_cart = large_element_coords - abe_coord;
    [phi, r] = cart2pol(relative_coords_cart(:,1), relative_coords_cart(:,2));
    phi = mod(phi, 2*pi); % 确保角度在 [0, 2*pi)
    r_large = zeros(M * K, 1);
    for m = 1:M
        phi_min = (m-1) * delta_phi;
        phi_max = m * delta_phi;

        % 找到属于该扇区且在 Rc 内的单元索引
        in_section_indices = find(phi >= phi_min & phi < phi_max & r <= Rc);

        if ~isempty(in_section_indices)
            % 获取这些单元的距离并排序
            r_in_section = r(in_section_indices);
            [sorted_r, sort_idx] = sort(r_in_section);

            % 取前 K 个，不足则用 inf 填充
            num_found = length(sorted_r);
            num_to_take = min(num_found, K);

            start_idx_in_vec = (m-1)*K + 1;
            end_idx_in_vec = start_idx_in_vec + num_to_take - 1;
            
            sorted_r = sorted_r / lambda;
            r_large(start_idx_in_vec : end_idx_in_vec) = 1 ./ sorted_r(1:num_to_take);
            % 其他默认为 0 (对应 1/inf)
        end
    end
    % 组合特征: 绝对位置 + 相对位置倒数
    ln = [abe_coord.' ./ [Lx; Ly]; r_relative_inv]; % 列向量形式
end

% --- 使用示例 ---
% 假设 N 是单元数, K, M, Rc 已定义
% all_ln = zeros(2 + K*M, N); % 预分配内存
% for n = 1:N
%     all_ln(:, n) = calculate_ABE_features(n, array_geometry_sample_i, K, M, Rc);
% end
