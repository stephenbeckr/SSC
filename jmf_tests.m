function [] = jmf_tests(in1, in2)
    
    %profile_run();
    %return
    
    if nargin >= 1
        plot_time_scaling_driver(in1, in2);
        return
    end
    run_time_scaling();

end

function [] = profile_run()
    profile off;
    clear all; % profile clear doesn't clear all the way or something... this seems to do it though.
    
    n   = 600;
    p   = 256;
    maxIter = 30;

    X   = randn(p,n);
    affine = true;
    lambdaE = 1; % bogus parameter

    profile clear;
    profile on;
    
    % ADMM
    % ---
    %alpha_rho = 10;
    %SSC_viaADMM(X, ...
    %    'lambda', lambdaE, 'maxIter', maxIter, 'printEvery', 1,...
    %    'tol', 1e-12, 'affine', affine, 'alpha_rho', alpha_rho, 'adaptiveRho', false,...
    %    'errHistEvery', maxIter+100, 'residHistEvery', maxIter+100);
    
    % TFOCS
    % ---
    tfocs_opts = struct( 'errFcn', @(objective,C) errFcn(reshape(C,n,n)) );
    tfocs_opts = struct();
    SSC_viaProxGradient(X, 'lambda', lambdaE,...
        'affine', affine, 'tfocs_opts', tfocs_opts , 'tol', 1e-6, 'maxIter', maxIter, ...
        'printEvery', 1);
    
    profile off


end

function [] = run_time_scaling()
    % Parameters for Section 4.2 of AAAI submission
    % (comes from Stephen's paper_deom_3_cont.m)
    % ---
    p = 256;     % ambient dimension 
    K = 10;      % number of subspaces 
    d = 3;       % dimension of each subspace 
    sigma = 0.1; % noise standard deviation
    rho_all = 200:50:500; % the rho used in the paper
    rho_all = round(linspace(20,50,5)); % bogus rho for quicker testing
    alpha   = [30,90];
    maxIter = 30;
    numExp  = 1; % number of trials to run

    ADMM_affine_times = zeros(numel(rho_all), numExp);
    TFOCS_affine_times = zeros(numel(rho_all), numExp);

    for densVal = 1 : numel(rho_all)
        rho = rho_all(densVal);
        N   = rho * K * d
        
        for trial = 1 : numExp
            TmpBasis = randn(p, p);
            [TmpBasis, ~] = qr(TmpBasis, 0);
            X = zeros(p,N);
            true_labels = zeros(N,1);
            for i = 1 : K % for each subspace 
                Ni = rho *d;
                ind = randperm(p,d);
                U  = TmpBasis(:,ind);
                X(:, (i-1)*Ni+1:i*Ni) = U * randn(d, Ni) + sigma*randn(p,Ni);
                true_labels((i-1)*Ni+1:i*Ni) = i; 
            end
            
            % Set up solver params
            % ---
            affine = true; % with affine constraint
            threshold = 1e-6;
            lambdaE = 1; % TODO JMF 16 Nov 2018: this is a bogus parameter; okay for timing though?
            
            % ADMM
            alpha_rho = 10; % TODO JMF 16 Nov 2018: this is a bogus parameter; okay for timing though?

            t_ = tic();
            SSC_viaADMM(X, ...
                'lambda', lambdaE, 'maxIter', maxIter, 'printEvery', 1,...
                'tol', 1e-12, 'affine', affine, 'alpha_rho', alpha_rho, 'adaptiveRho', false,...
                'errHistEvery', maxIter+100, 'residHistEvery', maxIter+100);
            
            ADMM_affine_times(densVal,trial) = toc(t_);

            % TFOCS
            %tfocs_opts = struct( 'errFcn', @(objective,C) errFcn(reshape(C,n,n)) );
            tfocs_opts = struct();
            
            t_ = tic();
            
            SSC_viaProxGradient(X, 'lambda', lambdaE,...
                'affine', affine, 'tfocs_opts', tfocs_opts , 'tol', threshold, 'maxIter', maxIter, ...
                'printEvery', 1);

            TFOCS_affine_times(densVal,trial) = toc(t_);

        end
    end

    save('time_scaling_data.mat');

end

function [] = plot_time_scaling_driver(in1, in2)
    if nargin == 1
        data_file = string(in1)
        plot_time_scaling_single(data_file);

    elseif nargin == 2
        data_file_baseline = string(in1);
        data_file_new = string(in2);
        plot_time_scaling_compare(data_file_baseline, data_file_new);

    else
        error('unhandled number of input arguments');
    end

end

function [] = plot_time_scaling_single(data_file)
    load(data_file);

    ADMM_affine_times_mean = mean(ADMM_affine_times, 2);
    TFOCS_affine_times_mean = mean(TFOCS_affine_times, 2);

    %TODO JMF 16 Nov 2018: plot std

    % Plot runtime scaling on log-log
    % ---
    N_vec = rho_all * K * d;

    figure(1);
    clf;
    hold on;

    plot(N_vec, ADMM_affine_times_mean, 'LineWidth', 2);
    plot(N_vec, TFOCS_affine_times_mean, 'LineWidth', 2);

    hold off;
    
    set(gca, 'xscale', 'log');
    set(gca, 'yscale', 'log');
    xlim([N_vec(1) N_vec(end)]);

    xlabel('number of data points');
    ylabel('running time (sec.)');

    legend('ADMM', 'TFOCS', 'Location', 'NorthWest');

end

function [] = plot_time_scaling_compare(data_file_baseline, data_file_new)
    S_baseline = load(data_file_baseline);
    S_new = load(data_file_new);

    % Some sanity checks
    % ---
    if ~all(S_baseline.rho_all == S_new.rho_all)
        error('rho_all vector doesn''t match in both files');
    end


    ADMM_affine_times_mean_baseline = mean(S_baseline.ADMM_affine_times, 2);
    TFOCS_affine_times_mean_baseline = mean(S_baseline.TFOCS_affine_times, 2);
    
    ADMM_affine_times_mean_new = mean(S_new.ADMM_affine_times, 2);
    TFOCS_affine_times_mean_new = mean(S_new.TFOCS_affine_times, 2);
    
    %TODO JMF 16 Nov 2018: plot std

    % Plot runtime scaling on log-log
    % ---
    N_vec = S_baseline.rho_all * S_baseline.K * S_baseline.d;

    figure(1);
    clf;
    hold on;

    plot(N_vec, ADMM_affine_times_mean_baseline, '-', 'LineWidth', 2);
    plot(N_vec, TFOCS_affine_times_mean_baseline, '-', 'LineWidth', 2);

    plot(N_vec, ADMM_affine_times_mean_new, '--', 'LineWidth', 2);
    plot(N_vec, TFOCS_affine_times_mean_new, '--', 'LineWidth', 2);

    hold off;
    
    set(gca, 'xscale', 'log');
    set(gca, 'yscale', 'log');
    xlim([N_vec(1) N_vec(end)]);

    xlabel('number of data points');
    ylabel('running time (sec.)');

    legend('ADMM (baseline)', 'TFOCS (baseline)', 'ADMM (new)', 'TFOCS (new)', 'Location', 'NorthWest');

end

