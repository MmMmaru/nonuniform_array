%% data preprocess
clear classes
dataFolder = 'dataset/planar_cp_dataset/';  % 你的数据目录
filename = sprintf(dataFolder + "Gn_%d", 500);
load(filename + ".mat");
points = model_points{1};

theta = -90:2:90;
phi = 0:10:180;
row = size(theta, 2);col = size(phi, 2);
f = 1;

N_elements = 16;

outC = 4;          % 目标图像通道数
outH = 64;         % 目标图像高度
outW = 64;         % 目标图像宽度

u = sind(theta).' * cosd(phi);
v = sind(theta).' * sind(phi);
u = reshape(u, [row col]); v = reshape(v, [row col]);
sample = 1:row*col;
[phi_grid, theta_grid] = meshgrid(phi, theta);
phi_grid = phi_grid(:);
phi_grid = repmat(phi_grid, 1, N_elements);
phi_grid = reshape(phi_grid, [row*col,1,1,N_elements]);
phi_grid = deg2rad(phi_grid);

nGridPoints = 64; % 网格精细度
u_grid_vec = linspace(-1, 1, nGridPoints);
v_grid_vec = linspace(-1, 1, nGridPoints);
[U_grid, V_grid] = meshgrid(u_grid_vec, v_grid_vec);


k = 1;
Gn_theta = G_complex_all(sample, 1, f, :);% [sample,2,1,k]
Gn_phi = G_complex_all(sample, 2, f, :);

Gn_lh = 1/sqrt(2) * (Gn_theta - 1j*Gn_phi) .* exp(-1j*phi_grid);
Gn_rh = 1/sqrt(2)*(Gn_theta + 1j*Gn_phi) .* exp(1j*phi_grid);
Gn = [];
Gn(:,1,:,:) = real(Gn_lh);
Gn(:,2,:,:) = imag(Gn_lh);
Gn(:,3,:,:) = real(Gn_rh);
Gn(:,4,:,:) = imag(Gn_rh);

order = [3,2,1];
Gn = permute(Gn, order); % [16 4 sample]
for j = 1:N_elements
    for c = 1:outC
        temp = Gn(j,c,:);
        temp = squeeze(temp);
        temp = reshape(temp, [row col]);
        G_(c,:,:) = griddata(u, v, temp, U_grid, V_grid, 'natural'); % 'natural' 是常用的插值方法
        idx_NAN = isnan(G_);
        G_(idx_NAN) = 0;
    end
    Y_set(k, :, :, :) = G_;
    k = k+1;
end
% predict AEP
K = 2; M = 16;
Rc_train = 1*15;
% ln_all = zeros(K*M + 2, N_elements);
X_set = [];
ln_all = [];
for j = 1:16
    % ln_all(:,j) = calculate_ABE_features_1(j, model_points{1}, K, M, Rc, 15, [Inf,Inf], 22.5, 22.5);
    ln_all(j,:,:) = calculate_xy_features(j, model_points{1}, 14, []);
end
X_set = [X_set; ln_all];
current_folder = "F:\PROJECT\nonuniform_array_prj\nonuniform_array"; 
py_folder = fullfile(current_folder, 'python_file');
if count(py.sys.path,py_folder) == 0
    insert(py.sys.path,int32(0),py_folder);
end
folder = "python_file\";
save(folder + "X_set.mat", "X_set");        
mod = py.importlib.import_module("predict_2dAEP");
py.importlib.reload(mod);   
py.predict_2dAEP.predict_2matlab();
% pyrunfile("predict_2dAEP.py");
load(folder + 'preds.mat'); 
%% show pred results
n = 5;
points = model_points{1};
figure (1);
scatter(points(:,1), points(:,2), 200, '.', 'b');
hold on;
scatter(points(n,1), points(n,2), 200, 'red');
hold off;
box on


Gn = [];
Gn(1,:,:) = YPred(n,3,:,:) + 1j*YPred(n,4,:,:);% rhcp
Gn(2,:,:) = YPred(n,1,:,:) + 1j*YPred(n,2,:,:);% lhcp

Gn(5,:,:) = Y_set(n,3,:,:) + 1j*Y_set(n,4,:,:);% rhcp
Gn(6,:,:) = Y_set(n,1,:,:) + 1j*Y_set(n,2,:,:);% lhcp

Gn = abs(Gn);
max1 = max(Gn(2,:,:), [], "all"	);% pred
max2 = max(Gn(6,:,:), [], "all"	);% sim

AEP_normed(1,:,:) = squeeze(10*log10(Gn(6,:,:)/max2));% s lhcp
AEP_normed(2,:,:) = squeeze(10*log10(Gn(2,:,:)/max1));% p lhcp
AEP_normed(3,:,:) = squeeze(10*log10(Gn(5,:,:)/max2));% s rhcp
AEP_normed(4,:,:) = squeeze(10*log10(Gn(1,:,:)/max1));% p rhcp

error_lhcp = max(10*log10(abs(Gn(6,:,:) - Gn(2,:,:)) / max2), [],"all")
error_rhcp = max(10*log10(abs(Gn(5,:,:) - Gn(1,:,:)) / max2), [],"all")

figure (2);
subplot(2,2,1);   
temp = squeeze(10*log10(Gn(6,:,:)/max2));
pcolor(U_grid,V_grid,temp);
xlabel('u');ylabel('v');
title('simmulated LHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
% axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-10, 0]);

subplot(2,2,2);
temp = squeeze(10*log10(Gn(2,:,:)/max1));
pcolor(U_grid,V_grid,temp);
xlabel('u');ylabel('v');
title('predicted LHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
% axis equal;
% axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-10, 0]);

subplot(2,2,3);
temp = squeeze(10*log10(Gn(5,:,:)/max2));
pcolor(U_grid,V_grid,temp);
xlabel('u');ylabel('v');
title('simmulated RHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
% axis equal;
% axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20, 0]);

subplot(2,2,4);
temp = squeeze(10*log10(Gn(1,:,:)/max1));
pcolor(U_grid,V_grid,temp);
xlabel('u');ylabel('v');
title('predicted RHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20, 0]);
%% array synthesis

thetaScanDeg = 0;
phiScanDeg = 0 ;
f0 = 20e9;
theta_scandeg = 0;
phi_scandeg = 0;
u0 = [0,0,0];
u0(1) = sind(theta_scandeg) * cosd(phi_scandeg);
u0(2) = sind(theta_scandeg) * sind(phi_scandeg);

F = zeros(8,outH,outW);
theta_expect = -90:0.5:90;
N = 16;
points = model_points{1};
freq = 20;% in Ghz
k0 = 2*pi * freq * 1e9 / 3e8;
Gn = [];
% F通道为[4,H,W]->E/theta,phi,rhcp,lhcp
for n = 1:N
    rn = [points(n,:)/1000,0];

    Gn(3,:,:) = YPred(n,3,:,:) + 1j*YPred(n,4,:,:);% RHCP
    Gn(4,:,:) = YPred(n,1,:,:) + 1j*YPred(n,2,:,:);% LHCP
    Gn(1,:,:) = 1/sqrt(2)*(Gn(3,:,:) + Gn(4,:,:));
    Gn(2,:,:) = 1/sqrt(2)*-1j*(Gn(3,:,:) - Gn(4,:,:));

    Gn(7,:,:) = Y_set(n,3,:,:) + 1j*Y_set(n,4,:,:);
    Gn(8,:,:) = Y_set(n,1,:,:) + 1j*Y_set(n,2,:,:);
    Gn(5,:,:) = 1/sqrt(2)*(Gn(7,:,:) + Gn(8,:,:));
    Gn(6,:,:) = 1/sqrt(2)*-1j*(Gn(7,:,:) - Gn(8,:,:));
%     Gn3 = gn_periodic;
    phase_n = k0 * (-dot(rn, u0));
    for i = 1:8
        G = squeeze(Gn(i,:,:));
        F(i,:,:) = squeeze(F(i,:,:)) + G.* exp(1j * k0 * (rn(1) * U_grid + rn(2) * V_grid) + 1j * phase_n);
    end
end

F(isnan(F)) = 0;
U = abs(F) .^2 /2 /377;
Gain = 4*pi*U/ N; 
Gain = 10 * log10(Gain);

tao = angle(F(8,:,:) ./ F(7,:,:))/2;
Ex = F(5,:,:) .* cos(tao) - F(6,:,:) .* sin(tao);
Ey = F(5,:,:) .* sin(tao) + F(6,:,:) .* cos(tao);
AR = abs(Ey ./ Ex);
idx = AR<1;
AR(idx) = 1 ./ AR(idx);
AR = squeeze(AR);

tao = angle(F(4,:,:) ./ F(3,:,:))/2;
Ex = F(1,:,:) .* cos(tao) - F(2,:,:) .* sin(tao);
Ey = F(1,:,:) .* sin(tao) + F(2,:,:) .* cos(tao);
AR_Pred = abs(Ey ./ Ex);
idx = AR_Pred<1;
AR_Pred(idx) = 1 ./ AR_Pred(idx);
AR_Pred = squeeze(AR_Pred);
idx = (U_grid.^2 + V_grid.^2)>1;
AR_Pred(idx) = NaN;


%     4. 绘图 (使用 pcolor)
subplot(3,3,1);
pcolor(U_grid, V_grid, AR);
xlabel('u');ylabel('v');
title('simmulated AR');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([0, 20]);

subplot(3,3,2);
pcolor(U_grid, V_grid, AR_Pred);
xlabel('u');ylabel('v');
title('predicted AR');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([0, 20]);

subplot(3,3,3);
temp = abs(AR-AR_Pred);
disp(max(temp,[],"all"));
pcolor(U_grid, V_grid, abs(AR-AR_Pred));
xlabel('u');ylabel('v');
title('AR error');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([0, 20]);

subplot(3,3,4);
pcolor(U_grid, V_grid, squeeze(Gain(8,:,:)));
xlabel('u');ylabel('v');
title('simmulated LHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20, 20]);

subplot(3,3,5);
pcolor(U_grid, V_grid, squeeze(Gain(4,:,:)));
xlabel('u');ylabel('v');
title('predicted LHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20, 20]);

subplot(3,3,6);
a = squeeze(Gain(4,:,:));b = squeeze(Gain(8,:,:));
% a(a<-10) = -10;b(b<-10) = -10;
temp = abs(a-b);
temp(temp == Inf) = 0;
disp(max(temp,[],"all"));
pcolor(U_grid, V_grid, temp);
xlabel('u');ylabel('v');
title('LHCP error');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([0,5]);

subplot(3,3,7);
pcolor(U_grid, V_grid, squeeze(Gain(7,:,:)));
xlabel('u');ylabel('v');
title('simmulated RHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20,10]);

subplot(3,3,8);
pcolor(U_grid, V_grid, squeeze(Gain(3,:,:)));
xlabel('u');ylabel('v');
title('predicted RHCP');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([-20, 10]);

subplot(3,3,9);
temp = abs(squeeze(Gain(3,:,:) - Gain(7,:,:)));
temp(temp == Inf) = 0;
disp(max(temp,[],"all"));  
pcolor(U_grid, V_grid, temp);
xlabel('u');ylabel('v');
title('RHCP error');
shading interp; % 平滑颜色过渡，隐藏网格线
axis equal;
axis([-1 1 -1 1]);
colormap('jet');
hcb = colorbar;
ylabel(hcb, 'dB');
clim([0,5]);