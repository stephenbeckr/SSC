function [] = test_proj_largest_k_affine()
    
    check_correctness()
    time_single_run()

end

function [] = check_correctness()
% Check correctness of proj_largest_k_affine_mex

    % An example from sparse subspace clustering (SSC)
    %rng(271828);
    n_rows = 700;
    n_cols = 600;
    k = 30;
    x = randn(n_rows, n_cols);
    lambda = randn();
    %lambda = randn(n_cols,1);
    zeroID = true;
    sparseOutput = false;
    
    % Make the prox operators
    proj_ref = @(x) proj_largest_k_affine(x, k, lambda, zeroID);
    proj_test = @(x) proj_largest_k_affine_mex(x, k, lambda, zeroID, sparseOutput);
    
    y_ref = proj_ref(x);
    y_test = proj_test(x);
    
    %y_ref
    %%y_test
    %full(y_test)

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
    lambda = randn(n_cols,1);
    zeroID = true;
    sparseOutput = true;

    % Make the prox operator
    proj = @(x) proj_largest_k_affine(x, k, lambda, zeroID);
    proj_largest_k_affine_mex(struct('num_threads', 8));
    proj = @(x) proj_largest_k_affine_mex(x, k, lambda, zeroID, sparseOutput);

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


function [x, S] = GSHP(b, K, lambda)
% [x, S] = GSHP(b, K, lambda)
% Computes a minimizer to the problem
%   min_x || x - b ||_2^2
% s.t.
%   x has at most k nonzeros
%   sum(x) == lambda
%
% GSHP stands for "Greedy Selector and Hyperplane Projector"
% This version does *not* allow for a weighted sum, but it is possible
%   to do that.
%
% This version DOES allow "b" to be a matrix
%   and in that case, the output "x" is a matrix, and the output
%   is equivalent to looping over the columns of x and b.
%   ("lambda" is the same for all columns, though this could easily
%    be changed if necessary)
%
% Stephen Becker, 2/217/2018
% Follows code from "Sparse Projections onto the Simplex"
%   by Kyrillidis, Becker, Ceverh, Kock, ICML 2013
%   Available at arXiv.org/abs/1206.1529
%   (In that algorithm, their "w" is our "b")

[~,j] = max( lambda*b ); % automatically vectorized over columns
S     = j;
nCols = size(b,2);
if nCols == 1
    for l = 2:K
        offset = (sum(b(S),1)-lambda)/(l-1);
        resid  = abs( b - offset );
        resid(S) = 0; % make sure we don't select an old index
        [~,j]   = max( resid );
        S       = sort([S;j]);
    end
    % final projection
    xS  = b(S) - ( sum(b(S)) - lambda)/K;
    x   = zeros(size(b,1),1);
    x( S ) = xS;

else
    for l = 2:K
        offset = zeros(1,nCols);
        for j = 1:nCols
            offset(j) = (sum(b(S(:,j),j),1)-lambda)/(l-1);
        end
        resid  = abs( b - offset );
        for j = 1:nCols
            resid(S(:,j),j) = 0;
        end
        [~,j]   = max( resid );
        S       = [S;j];
    end
    % final projection
    x   = zeros(size(b,1),nCols);
    for j = 1:nCols
        SS  = S(:,j);
        xS  = b(SS,j) - ( sum(b(SS,j)) - lambda)/K;
        x( SS, j ) = xS;
    end
end
end

function [X] = proj_largest_k_affine(X, k, lambda, zeroID)
% Wrapper around GSHP

if nargin < 3
    zeroID = true;
end

[nRows,nCols] = size(X);
if nRows < nCols
    error('Cannot enforce diag(Y) == 0 if nCols > nRows');
end

if numel(lambda) > 1 && numel(lambda) ~= nCols
    error('lambda should have nCols elements');
end

if numel(lambda) == 1
    lambda = lambda*ones(nCols,1);
end

if zeroID && nCols > 1
    for col = 1:nCols
        ind     = [1:col-1,col+1:nRows];
        x       = X(ind,col);
        x       = GSHP(x, k, lambda(col)); % project to be sparse
        if nnz(x) > k, error('Did not project correctly!'); end
        X(col,col)  = 0;
        X(ind,col)  = x;
    end
else
    for col = 1:nCols
        x       = X(:,col);
        x       = GSHP(x, k, lambda(col)); % project to be sparse
        if nnz(x) > k, error('Did not project correctly!'); end
        X(:,col)  = x;
    end
end

end
