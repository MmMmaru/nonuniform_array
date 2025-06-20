function [gn_complex_single, angles_uv] = get_AEP_HFSS(excite_element, u0, points, f)
    % !!! 概念性代码，需要根据您的 HFSS 版本和设置调整 !!!
    % !!! 需要对 HFSS 脚本 API 有深入了解 !!!
    % 输入为mm
    N = size(points, 1);
    lambda = 0.3/f;
    k0 = 2*pi/lambda;
    try
        % 连接到 HFSS (可能需要 HFSS 正在运行)
        hHfss = actxserver('AnsoftHFSS.HfssScriptInterface'); % 类名可能因版本而异
        hDesktop = hHfss.GetAppDesktop();
        hProject = hDesktop.GetActiveProject(); % 获取活动项目
        hDesign = hProject.GetActiveDesign();  % 获取活动设计
        hReport = hDesign.GetModule('ReportSetup');
        hModule = hDesign.GetModule("Solutions");
    
        % 更改激励
        editSourcesArg = cell(1, N + 1);
    
        % Element 1: Global Settings (inner cell array)
        % Using logical(0) for VBScript's 'false'
        globalSettings = {'IncludePortPostProcessing:=', logical(0), 'SpecifySystemPower:=', logical(0)};
        editSourcesArg{1} = globalSettings;
    
        fprintf('Preparing source arguments...\n');
        % Elements 2 to N+1: Source definitions (inner cell arrays)
        for i = 2:(N + 1) % Loop matches VBScript loop (index i corresponds to port i:1)
            elementIndex = i - 1; % Corresponding index in your 'points' matrix (1 to N)
    
            % Calculate phase for this element based on position
            rn = [points(elementIndex, :) / 1000, 0]; % Convert mm to meters, assume z=0
            phase_n_rad = k0 * (-dot(rn, u0));
            phase_n_deg = rad2deg(phase_n_rad);
    
            % Determine magnitude based on VBScript logic
            if i == excite_element+1 % First source in the loop (port "2:1")
                magStr = '1W'; % Use Watts as in VBScript
                phaseStr = '0deg'; % Use 0deg as in VBScript
            else      % Subsequent sources (ports "3:1" onwards)
                magStr = '0W'; % Use 0W as in VBScript
                phaseStr = sprintf('%fdeg', phase_n_deg); % Use calculated phase
            end
    
            % Construct the source name string "i:1"
            portNameStr = sprintf('%d:1', i);
    
            % Create the cell array for this source's settings
            sourceSettings = { ...
                'NAME:=', portNameStr, ...
                'Magnitude:=', magStr, ...
                'Phase:=', phaseStr ...
            };
    
            % Assign this source's settings to the main argument cell array
            editSourcesArg{i} = sourceSettings; % Index matches VBScript Array element index
        end
        invoke(hModule, 'EditSources', editSourcesArg);
        
        % --- 使用 HFSS API 命令 ---
        reportName = 'gn'; % 您在 HFSS 中创建的报告名称
        exportPath = fullfile(pwd, 'gn.csv'); % 导出文件的完整路径
    
        % 调用 HFSS API 导出文件 (具体命令需查阅 HFSS 文档)
        hReport.ExportToFile(reportName, exportPath, false); % false 可能表示不覆盖
        % 或者：hReport.ExportToFile(reportName, exportPath); % 较新版本可能不同
    
        fprintf('尝试通过 COM 导出报告 %s 到 %s\n', reportName, exportPath);
        % 确保 HFSS 有足够时间完成导出，可能需要加 pause
    
        % --- 导出成功后，读取文件 ---
        if exist(exportPath, 'file')
            patternTable = readtable(exportPath); % 使用 readtable 或 readmatrix
            % ... 处理数据 ...
            fprintf('成功读取自动导出的文件。\n');
        else
            warning('未能找到导出的文件 %s。', exportPath);
        end
    
        release(hHfss); % 或其他清理方式
    
    catch ME
        warning('连接 HFSS 或执行 COM 命令时出错:');
        disp(ME.message);
        % 可以在这里添加更详细的错误处理
    end
    
    
    % 假设数据已加载到 patternTable
    % patternTable = readtable('your_exported_data.csv', opts);
    
    % 检查列名 (根据您实际导出的文件修改)
    disp('Available columns:');
    disp(patternTable.Properties.VariableNames);
    
    freq_col_name = 'Freq_GHz_'; % 修改为实际频率列名
    phi_col_name = 'Phi_deg_';       % 修改为实际 Phi 列名
    theta_col_name = 'Theta_deg_';     % 修改为实际 Theta 列名
    mag_col_name = 'mag_rEX__V_'; % !! 修改为实际幅度列名 (如 GainTotal, rETotal_mag 等)
    phase_col_name = 'ang_deg_rEX__deg_'; % !! 检查是否存在相位列，修改为实际相位列名
    
    % --- !!! 关键：检查是否存在相位数据 !!! ---
    if ~ismember(phase_col_name, patternTable.Properties.VariableNames)
        error(['错误：找不到相位数据列 "%s"。', ...
               'calculate_virtual_excitations 函数需要复数方向图（幅度和相位）。', ...
               '请返回 HFSS，确保导出报告时包含了相位信息（例如 "Phase" 或 "cGainTotal" 等），然后重新导出并加载数据。'], ...
               phase_col_name);
    end
    
    fprintf('找到了幅度和相位列。\n');
    
    % 选择您感兴趣的频率点
    target_frequency = f; % 例如 10 GHz, 修改为您需要的值
    
    % 找到该频率对应的行索引
    freq_indices = abs(patternTable.(freq_col_name) - target_frequency) < 1e-6; % 使用容差比较频率
    
    if ~any(freq_indices)
        error('错误：在数据中找不到目标频率 %g Hz。请检查导出的频率点。', target_frequency);
    end
    
    % 提取该频率下的数据
    selected_data = patternTable(freq_indices, :);
    
    % 提取角度和幅度/相位
    phi_deg = selected_data.(phi_col_name);
    theta_deg = selected_data.(theta_col_name);
    magnitude = selected_data.(mag_col_name); % 幅度 (可能是 dB 或线性值)
    phase_deg = selected_data.(phase_col_name); % 相位 (可能是度)
    
    fprintf('已提取频率 %g GHz 的数据，共 %d 个角度点。\n', target_frequency, height(selected_data));
    
    % --- 幅度转换 (假设 HFSS 导出的是 dB) ---
    % 如果导出的是线性值(如 rETotal)，则不需要这步或进行其他转换
    % 检查列名或 HFSS 设置确认单位
    is_db = contains(mag_col_name, 'dB', 'IgnoreCase', true);
    if is_db
        fprintf('假设幅度 "%s" 是 dB 单位，转换为线性值...\n', mag_col_name);
        magnitude_linear = 10.^(magnitude / 20); % 幅度转换为线性值 (若是功率 dB，则除以 10)
    else
        fprintf('假设幅度 "%s" 是线性单位。\n', mag_col_name);
        magnitude_linear = magnitude;
    end
    
    % --- 相位转换 (假设 HFSS 导出的是度) ---
    phase_rad = deg2rad(phase_deg);
    
    % --- 计算复数方向图 gn_complex ---
    % 这个 gn_complex 代表 *当前加载文件* 对应的那个天线单元在该频率下的 AEP
    % !! 注意：这只是单个单元的数据 !!
    gn_complex_single = magnitude_linear .* exp(1j * phase_rad);
    
    % 确保是列向量
    gn_complex_single = gn_complex_single(:);% 转换角度为弧度
    phi_rad = deg2rad(phi_deg(:));   % 确保是列向量
    theta_rad = deg2rad(theta_deg(:)); % 确保是列向量
    
    % 计算 u, v 坐标
    u = sind(theta_deg(:)) .* cosd(phi_deg(:));
    v = sind(theta_deg(:)) .* sind(phi_deg(:));
    
    % 组合成 angles_uv 矩阵 [N_angles x 2]
    angles_uv = [u, v];
    rn = [points(excite_element, :) / 1000, 0];
    gn_complex_single = gn_complex_single .* exp(-1j * k0 * (rn(1) * u + rn(2) * v));
    fprintf("success\n");
end
