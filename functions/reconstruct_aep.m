function predicted_aep = reconstruct_aep(predicted_cn_complex, P, Q, d, angles_uv, g_ave_complex)
    N_angles = size(angles_uv, 1);
    N_virtual = P * Q;

    % 构建虚拟阵列坐标 (同上)
    p_indices = -(P-1)/2 : (P-1)/2;
    q_indices = -(Q-1)/2 : (Q-1)/2;
    [q_grid, p_grid] = meshgrid(q_indices, p_indices);
    virtual_coords = [p_grid(:)*d, q_grid(:)*d]; % [N_virtual x 2]

    % 构建方向矢量 E_vec (类似 E, 但用于单个 ABE 重构)
    u = angles_uv(:, 1);
    v = angles_uv(:, 2);
    E_vec = exp(1j * 2*pi*(u * virtual_coords(:,2).' + v * virtual_coords(:,1).')); % [N_angles x N_virtual]

    % 计算虚拟子阵因子
    virtual_array_factor = E_vec * predicted_cn_complex; % [N_angles x 1]

    % 乘以平均方向图 (Equation 7)
    predicted_aep = g_ave_complex .* virtual_array_factor;
end