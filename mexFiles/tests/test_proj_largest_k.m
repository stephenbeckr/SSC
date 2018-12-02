function [] = test_proj_largest_k()
    
    check_correctness()
    %time_single_run()

end

function [] = check_correctness()
% Check correctness of proj_largest_k_mex

    % An example from sparse subspace clustering (SSC)
    n_rows = 7;
    n_cols = 6;
    k = 3;
    x = randn(n_rows, n_cols);
    zeroID = true;
    sparseOutput = false;
    
    % Make the prox operators
    proj_ref = @(x) proj_largest_k(x, k, zeroID);
    proj_test = @(x) proj_largest_k_mex(x, k, zeroID, sparseOutput);
    
    y_ref = proj_ref(x);
    y_test = proj_test(x);
    
    %y_ref
    %y_test

    rel_err = norm(y_ref - y_test, 'fro') / norm(y_ref, 'fro');
    fprintf('relative error = %1.5e\n', rel_err);

end

function [] = time_single_run()
% Time prox_l1_mat implementations

    minimum_runtime = 1;

    % An example from sparse subspace clustering (SSC)
    n_rows = 700;
    n_cols = 600;
    k = 10;
    x = randn(n_cols, n_cols);
    zeroID = true;
    sparseOutput = false;

    % Make the prox operator
    proj = @(x) proj_largest_k(x, k, zeroID);
    proj_largest_k_mex(struct('num_threads', 4));
    proj = @(x) proj_largest_k_mex(x, k, zeroID, sparseOutput);

    % Warm up
    n_done = 0;
    t_ = tic();
    while true
        if toc(t_) >= minimum_runtime
            break
        end

        y = proj(x);
        n_done = n_done + 1;
    end

    % Measure runtime
    times = zeros(n_done,1);
    for n=1:n_done
        t_ = tic();
        y = proj(x);
        times(n) = toc(t_);
    end

    fprintf('proj min/mean/max runtime = %1.5e  %1.5e  %1.5e  seconds\n', min(times), mean(times), max(times));

end


function y = findLargestK( x, K )
% Returns a closest point y to x, in Euclidean distance,
%   such that y is only K nonzero entries.

% you could adapt this easily if you wanted, but now just assume vector
% input
if length(x) < numel(x), error('not designed for matrices, assumes x is a vector'); end


y        = zeros(size(x));
if K > 4*log( length(x) )
    % better to do a sort, for log(n)*n complexity
    % (this is not precise, since the constants are unknown)
    [~,ind] = sort( abs(x), 'descend' );
    y( ind(1:K) ) = x( ind(1:K) );
else
    % better to loop K times, for K*n complexity
    xa = abs(x);
    for k = 1:K
        [~,ind] = max(xa);
        xa(ind)=0; % don't select it again
        y( ind ) =x( ind );
    end
end
end

function [X] = proj_largest_k(X, k, zeroID)
% Wrapper around findLargestK

if nargin < 3
    zeroID = true;
end

[nRows,nCols] = size(X);
if nRows < nCols
    error('Cannot enforce diag(Y) == 0 if nCols > nRows');
end

if zeroID && nCols > 1
    for col = 1:nCols
        ind     = [1:col-1,col+1:nRows];
        x       = X(ind,col);
        x       = findLargestK(x, k); % project to be sparse
        if nnz(x) > k, error('Did not project correctly!'); end
        X(col,col)  = 0;
        X(ind,col)  = x;
    end
else
    for col = 1:nCols
        x       = X(:,col);
        x       = findLargestK(x, k); % project to be sparse
        if nnz(x) > k, error('Did not project correctly!'); end
        X(:,col)  = x;
    end
end

end
