function cn = calculate_virtual_excitations(gn_complex, P, Q, d, angles_uv, g_ave_complex)
    % gn_complex: 单个 ABE 的复数 AEP 在采样角度上的向量 [N_angles x 1]
    % P, Q: 虚拟子阵列的维度 (奇数)
    % d: 虚拟子阵列的单元间距 (in wavelengths)
    % angles_uv: AEP 采样角度 [N_angles x 2], 每行是 [uk, vk]
    % g_ave_complex: 所有 AEP 的平均值在采样角度上的向量 [N_angles x 1]
    % cn [N_virtual x 1]
    
    N_angles = size(angles_uv, 1);
    N_virtual = P * Q; % 虚拟单元总数

    % 构建虚拟阵列坐标 (中心在原点)
    p_indices = -(P-1)/2 : (P-1)/2;
    q_indices = -(Q-1)/2 : (Q-1)/2;
    [q_grid, p_grid] = meshgrid(q_indices, p_indices);
    virtual_coords = [p_grid(:), q_grid(:)]; % [N_virtual x 2]

    % 构建矩阵 E (Equation 11)
    u = angles_uv(:, 1);
    v = angles_uv(:, 2);
    E = exp(1j * 2 * pi * d*(virtual_coords(:,1) * v.' + virtual_coords(:,2) * u.')).'; % [N_angles x N_virtual] 注意 u,v 和 x,y 的对应关系

    % 构建矩阵 G (Equation 10) - 对角矩阵
    G = diag(g_ave_complex); % [N_angles x N_angles]

    % 构建矩阵 Z (Equation 9)
    Z = G * E; % [N_angles x N_virtual]

    % 解最小二乘问题 (Equation 14)
    % 使用 pinv (伪逆) 更稳定，或者直接计算
    cn = Z \ gn_complex; % [N_virtual x 1]
%     cn = (Z' * Z) \ (Z' * gn_complex); % 更推荐用反斜杠

end

% --- 使用示例 ---
% 假设 P, Q, d, angles_uv 已定义
% g_all_complex = cell2mat(simulated_AEPs_sample_i); % 将cell转为矩阵 [N_angles x N]
% g_ave_complex = mean(g_all_complex, 2); % 计算平均 AEP [N_angles x 1]

% all_cn = zeros(P*Q, N); % 预分配内存
% for n = 1:N
%     gn_complex = g_all_complex(:, n);
%     all_cn(:, n) = calculate_virtual_excitations(gn_complex, P, Q, d, angles_uv, g_ave_complex);
% end

% 提取目标值 (幅度和相位)
% target_mag = abs(all_cn);
% target_phase = angle(all_cn);