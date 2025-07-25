
i = 1;
f = 1;
Gn_theta = G_complex_all(:,1,1,i);
Gn_phi = G_complex_all(:,2,1,i);
Gn_rhcp = 1/sqrt(2)*(Gn_theta + 1j*Gn_phi);
Gn_rhcp = squeeze(Gn_rhcp);
% Gn_rhcp = reshape()
theta_points = size(Gn_rhcp, 1);
theta = linspace(-90,90, theta_points);

n = 2; % 分子阶数 (Numerator order of Padé)
d = 2; % 分母阶数 (Denominator order of Padé)
k = 7; % 空间多项式阶数 (Spatial polynomial order)

% 采样点 (Sampling Points)
% 根据论文选择采样点
freq_samples = 1; % MHz
theta_samples_deg = linspace(-90,90,181); % degrees

% 将单位转换为SI
theta_samples = deg2rad(theta_samples_deg); % radians
s_samples = 1i * 2 * pi * freq_samples; % 复频率 s = jw


Nf = length(freq_samples);
N_theta = length(theta_samples);
N_total_samples = Nf * N_theta;
num_unknowns = (n + 1) * (k + 1) + d * (k + 1);

A = zeros(N_total_samples, num_unknowns);
b = zeros(N_total_samples, 1);

% 构造 theta 的多项式基向量
theta_basis = @(theta_val) theta_val.^(0:k);

% size(F_) = [Nf, N_theta]
F_true_samples = reshape(Gn_rhcp, Nf, N_theta);

row_idx = 1;
for i = 1:Nf
    for j = 1:N_theta
        s_val = s_samples(i);
        theta_val = theta_samples(j);
        F_val = F_true_samples(i, j);
        
        % 当前角度的多项式基
        T_basis = theta_basis(theta_val); % 1x(k+1) vector
        
        % --- 填充A矩阵的行 ---
        
        % Part 1: 分子系数 N_i^p
        % N(theta, s) = [N0(theta), N1(theta), ... Nn(theta)] * [1, s, ..., s^n]'
        col_start = 1;
        for p = 0:n % for each s^p
            A(row_idx, col_start : col_start + k) = T_basis * (s_val^p);
            col_start = col_start + (k + 1);
        end
        
        % Part 2: 分母系数 D_j^p
        % D(theta, s) - s^d
        for p = 0:d-1 % for each s^p
             A(row_idx, col_start : col_start + k) = -F_val * T_basis * (s_val^p);
             col_start = col_start + (k + 1);
        end
        
        % --- 填充b向量 ---
        b(row_idx) = F_val * (s_val^d);
        
        row_idx = row_idx + 1;
    end
end

coeffs = A\b;
