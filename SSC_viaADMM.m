function [C,errHist,resid,objective,parameters] = SSC_viaADMM(X, varargin )
% C = SSC_viaADMM( X )
%   solves the Sparse Subspace Clustering problem of Vidal et al.
%   using the ADMM algorithm. i.e., solves
%
% min_{C}  lambda/2|| X - X*C ||_F^2 + ||C||_1
%   s.t. diag(C) = 0
%    and (if 'affine' is true), ones(n,1)*C = ones(n,1)
%
%   where X is a p x n data matrix
% 
% C = SSC_viaADMM( X, 'parameter', value, ... ) allows for extra parameters
% such as:
%   'lambda'        regularization parameter; by default, alpha_lambda*mu
%       (where mu is a coherence parameter; for alpha_lambda<1, C=0 is
%       optimal );
%   'alpha_lambda'  See above
%   'rho'           ADMM parameter; any rho>0 works, but some lead to
%       faster convergence. Default: rho = alpha_rho*alpha_lambda;
%   'alpha_rho'     See above
%   'maxIter'       Max number of iterations (default: 200 )
%   'errFcn'        An optional user-supplied error function, evaluated
%       every iteration
%   'printEvery'    How often to display to the screen (between 1 and inf)
%   'affine'        Include ones(n,1)*C=ones(n,1) constraint (default: false)
%   'tol'           Stopping tolerance (default: 2e-4 ), based on the 
%       absolute l_inf norm error between the two primal ADMM variables
%   'errHistEvery'  Computes l_inf norm between two primal ADMM variables,
%       and any user supplied errFcn, every errHistEvery iterations
%       (default: 1 )
%   'residHistEvery' Computes residual and objective value every
%       residHistEvery iterations (default: 1)
%
%   'adaptiveRho'   Updates rho according to the 2016 Adaptive ADMM paper
%       of Zheng Xu, Mario?Figueiredo, and Tom Goldstein
%   'T_f'           How often to update rho, if adaptiveRho is turned on
%   'epsCurvature'  Parameter (default: 0.2) for adaptiveRho algorithm
%
% [C,errHist,resid,objective,parameters] = SSC_viaADMM( X, ... )
%   returns output information. errHist(:,1) is the l_inf norm error
%   between the two primal ADMM variables; errHist(:,2) is the errFcn
%   output (if supplied);  resid(:) is || X - X*C ||_F,
%   while objetive(:) is lambda/2|| X - X*C ||_F^2 + ||C||_1.
%
% Stephen Becker and Farhad Pourkamali-Anaraki 2018
%   https://github.com/stephenbeckr/SSC
% Based off algorithm of Ehsan Elhamifar and Rene Vidal 2012, but adding
%   in an O(n^2) matrix inversion instead of O(n^3)
%   Elhamir and Vidal paper: https://arxiv.org/abs/1203.1005
%
% Adaptive ADMM ("AADMM") is from the paper
% "Adaptive ADMM with Spectral Penalty Parameter Selection" (2016)
% by ?Zheng Xu, Mario?Figueiredo, and Tom Goldstein
% http://proceedings.mlr.press/v54/xu17a/xu17a.pdf


param  = inputParser;
addParameter( param, 'maxIter', 200, @(m) (m>=1) );
addParameter( param, 'errFcn', [] ); % e.g., @(Z) evalSSR_error( Z, true_labels );
addParameter( param, 'errFcn_uses_Z', true ) % use errFcn(Z) if true, or errFcn(C) if false
addParameter( param, 'printEvery', 10 );
addParameter( param, 'tol', 2e-4 ); % stopping tolerance
addParameter( param, 'affine', false );
addParameter( param, 'lambda', [], @(l) (l>0) );
addParameter( param, 'rho', [], @(rho) (rho>0) );
addParameter( param, 'alpha_lambda', 800, @(a) (a>=1) );
addParameter( param, 'alpha_rho', 1 );
addParameter( param, 'errHistEvery', 1 ); % how often to compute errFcn and 
addParameter( param, 'residHistEvery', 1 ); % how often to compute residual and objective

addParameter( param, 'adaptiveRho', false ); % turns on the "AADMM" algorithm
addParameter( param, 'T_f', 2, @(tf) (tf>=0) ); % for AADMM, this is how often to re-estimate rho
addParameter( param, 'epsCurvature', 0.2 ); % quality threshold for curvature estimate
% (In the AADMM paper, what we call "rho", they call "tau")

parse(param,varargin{:});
parameters  = param.Results;
maxIter     = parameters.maxIter;
errFcn      = parameters.errFcn;
errFcn_uses_Z = parameters.errFcn_uses_Z;
printEvery  = parameters.printEvery; if isinf(printEvery), printEvery=0; end
tol         = parameters.tol;
affine      = parameters.affine;
lambda      = parameters.lambda;
rho         = parameters.rho;
alpha_lambda= parameters.alpha_lambda;
alpha_rho   = parameters.alpha_rho;
errHistEvery= parameters.errHistEvery; if isinf(errHistEvery), errHistEvery = 0; end
residHistEvery= parameters.residHistEvery; if isinf(residHistEvery), residHistEvery = 0; end
adaptiveRho = parameters.adaptiveRho;
T_f         = parameters.T_f;
epsCurvature= parameters.epsCurvature;
if ~adaptiveRho
    T_f = 0; % turns off this feature
end

[p,n]   = size(X);

% Use conventions of Ehsan Elhamifar and Rene Vidal's 2012 paper
XtX     = X'*X; % n x n
if isempty( lambda )
    temp    = abs( XtX - diag(diag(XtX)) );
    mu_correlation = min(max(temp)); % eq (11) in our arXiv paper
    lambda  = alpha_lambda/mu_correlation; % "lambda_E" in arXiv paper
    parameters.lambda = lambda;
end
if isempty( rho )  % ADMM parameter, only affects convergence rate
    if isempty( alpha_rho )
        alpha_rho = 1;
    end
    rho     = alpha_rho * alpha_lambda;
    parameters.rho = rho;
    parameters.alpha_rho = alpha_rho;
end
XtX     = lambda*XtX;

% ==== precomputation ====
returnProxH(); % zero-out any persistent memory
Afun    = returnProxH( X, lambda, rho, affine );

% And soft-thresholding operation
parameters.usedFastShrinkage = false;
softThresh  = @(X, t) sign(X).*max( 0, abs(X) - t );
if 2==exist('tfocs_where','file')
    if 3~=exist('shrink_mex','file')
        addpath( fullfile( tfocs_where, 'mexFiles' ) );
    end
    if 3==exist('shrink_mex','file')
        softThresh  = @(x,t) shrink_mex(x,t); 
        parameters.usedFastShrinkage = true;
    else
        warning('SSC:mexNotCompiled',...
            'Found shrink_mex.c but not compiled; we suggest trying to compile it. You can turn this warning off');
        % to turn warning off, try   >  warning('off','SSC:mexNotCompiled');
    end
end


C       = zeros(n);
Z       = zeros(n); % appropriate?
Dual1   = zeros(n);
if affine, Dual2 = zeros(n,1); end
errHist = zeros(maxIter, 1 + isempty( errFcn ) );
[resid,objective]   = deal( zeros(maxIter,1) );
% ==== main ADMM loop ====
for k = 1:maxIter
    
    if adaptiveRho
        C_old   = C;
        Z_old   = Z;
    end
    
    if affine
        temp = XtX + rho*( C - Dual1/rho + ones(n,1)*(ones(1,n)-Dual2'/rho) );
    else
        temp = XtX + rho*( C - Dual1/rho );
    end
    Z   = Afun( temp );
    %Z   = Z - diag(diag(Z));
    %Z(1:(n+1):end) = 0; % faster version of the above % is this OK?? No.
    % Aug 2018, commenting out the above. We should not enforce diag(Z)=0
    % or else it messes up the ADMM algorithm.
    
    C   = softThresh( Z + Dual1/rho, 1/rho );
%     C   = C - diag(diag(Z));
    C(1:(n+1):end) = 0; % faster version of the above
    
    % Possibly update rho for AADMM
    if ~mod( k, T_f )
        Dual1hat   = Dual1 + rho*( Z - C_old );
        if affine, Dual2hat = Dual2 + rho*( Z'*ones(n,1) - ones(n,1) ); end
    end
    
    % Update Lagrange Multipliers
    Dual1   = Dual1 + rho*( Z - C );
    if affine, Dual2 = Dual2 + rho*( Z'*ones(n,1) - ones(n,1) ); end
    
    % Possibly update rho for AADMM
    if ~mod( k, T_f )
        % The very first time, do nothing, just save variables
        if k > T_f
            dLambda     = Dual1hat - Dual1hat_old;
            if affine, dLambda = [dLambda(:); Dual2hat - Dual2hat_old]; end
            
            % dH = A( Z - Z_old ) where "A" is the abstract linear operator
            dH          = -( Z - Z_old );
            if affine, dH   = [ dH(:); dH'*ones(n,1) ]; end
            
            % dG = B( C - C_old ), where B(C) = C - diag(C), but
            %   since diag(C)=0 is already enforced, we can remove this.
            dG          = C - C_old;
            if affine, dG   = [ dG(:); zeros(n,1) ]; end
            
            rho         = updateRho( rho, dLambda, dH, dG, epsCurvature ); % AADMM paper
            Afun        = returnProxH( X, lambda, rho, affine );
        end
        Dual1hat_old    = Dual1hat;
        if affine, Dual2hat_old = Dual2hat; end
    end
    
    if k==1 || ~mod( k, errHistEvery )
        errHist(k,1)    = norm( Z(:) - C(:), Inf );
        if affine, errHist(k,1) = errHist(k,1) + norm( ones(1,n)*Z - ones(1,n), Inf ); end
        if ~isempty( errFcn )
            if errFcn_uses_Z
                errHist(k,2)    = errFcn( Z );
            else
                errHist(k,2)    = errFcn( C );
            end
        end
    else
        % for large problems, may not want to compute error every iteration
        errHist(k,:) = errHist(k-1,:);
    end
    
    if k==1 || ~mod( k, residHistEvery )
        resid(k)    = norm( X - X*Z, 'fro' );
        objective(k)= norm(Z(:),1) + lambda/2*resid(k)^2;
    else
        resid(k)    = resid(k-1);
        objective(k)= objective(k-1);
    end
    
    do_break = ( errHist(k,1) < tol );
    if ~mod( k, printEvery ) || do_break || (printEvery >0 && k==maxIter )
        fprintf('Iter %4d, ADMM residual %.2e, objective %.2e', k, errHist(k,1), objective(k) );
        if ~isempty( errFcn )
            fprintf(', errFcn %.2e', errHist(k,2) );
        end
        fprintf('\n');
    end
    if do_break
        fprintf('Reached tolerance, ending\n');
        errHist     = errHist(1:k,:);
        resid       = resid(1:k,:);
        objective   = objective(1:k,:);
        break
    end
end
end % end of main routine



% == Subroutines ==
function Afun = returnProxH( X, lambda, rho, affine )
 persistent XXt
 if nargin == 0
     XXt = []; % do this to save memory
     return
 end
 if isempty( XXt )
     XXt = X*X';
 end
 [p,n]   = size(X);
 if affine
     % Make an operator that does the equivalent of
     %   @(x) inv(lambdaE*(X'*X)+rho*eye(n)+rho*ones(n,n)) * x
     % but do it efficiently
     
     mm1     = sqrt(lambda);
     mm2     = sqrt(rho);
     
     Y1      = sqrt(lambda)*sqrt(rho)*(X*ones(n,1));
     YYt     = [ lambda*(XXt), Y1; ...
         Y1', rho*n ];
     %     iYYt    = inv( eye(p+1) + 1/rho*YYt );
     % make it implicit
     %     Afun    = @(RHS) RHS/rho - ([mm1*X',mm2*ones(n,1)]*(iYYt*[mm1*(X*RHS);mm2*(ones(1,n)*RHS)]) )/(rho^2);
     
     % Alternatively, a bit more stable
     rChol   = chol( eye(p+1) + 1/rho*YYt );
     iYYt    = @(RHS) rChol\( rChol'\RHS);
     Afun    = @(RHS) RHS/rho - ([mm1*X',mm2*ones(n,1)]*(iYYt([mm1*(X*RHS);mm2*(ones(1,n)*RHS)])) )/(rho^2);
     
 else
     % Make an operator that does the equivalent of
     %   @(x) inv(lambdaE*(X'*X)+rho*eye(N)) * x
     % but do it efficiently
     %     A   = inv(XtX+rho*eye(n));
     %     Afun    = @(RHS) A*RHS;
     
     %     iYYt    = inv( eye(p) + 1/rho*lambda*(X*X') );
     %     Afun    = @(RHS) RHS/rho - (lambda/(rho^2))*(X'*(iYYt*(X*RHS)) );
     
     % Alternatively, a bit more stable,
     rChol   = chol( eye(p) + 1/rho*lambda*(XXt) );
     iYYt    = @(RHS) rChol\( rChol'\RHS);
     Afun    = @(RHS) RHS/rho - (lambda/(rho^2))*(X'*(iYYt(X*RHS)) );
 end
end


function rho = updateRho( oldRho, dL, dH , dG, epsCurvature)
 % Update rho (known as tau in the paper) according to the AADMM paper
 %  of Xu, Figueiredo, Goldstein 2016
 %
 % dL referes to dLambda
 % dL = lambda_k - lambda_{k_0}
 % dH = A( u_k - u_{k_0} )
 % dG = B( v_k - v_{k_0} )
 %
 % Note: all variables should have "hats" on them, which I left off
 %  to keep notation cleaner, but be aware that the paper
 %  uses conventions like \hat{\beta} = 1/\beta, so the hats are important
 %  if you translate back to the paper.
 
 % Eq (26) to find the "steepest descent" SD and minimum gradient "MG"
 %  stepsizes.
 dHdL = dH(:)'*dL(:);
 dLdL = norm(dL,'fro')^2;
 dHdH = norm(dH,'fro')^2;
 aSD  = dLdL / dHdL;
 aMG  = dHdL / dHdH;

 % Eq (27)
 if 2*aMG > aSD
     a  = aMG;
 else
     a  = aSD - aMG/2;
 end
 
 % Repeat for dG
 dGdL   = dG(:)'*dL(:);
 dGdG   = norm(dG,'fro')^2;
 bSD    = dLdL / dGdL;
 bMG    = dGdL / dGdG;
 % Eq (28)
 if 2*bMG > bSD
     b  = bMG;
 else
     b  = bSD - bMG/2;
 end
 
 % Now, test the correlations, Eq (29)
 aCorr  = dHdL / sqrt( dHdH*dLdL );
 bCorr  = dGdL / sqrt( dGdG*dLdL );
 
 % Eq (30)
 if aCorr > epsCurvature
     if bCorr > epsCurvature
         rho    = sqrt( a*b );
     else
         rho    = a;
     end
 else
     if bCorr > epsCurvature
         rho    = b;
     else
         rho    = oldRho;
     end
 end
end % end of subroutine
