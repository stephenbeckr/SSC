function makeMex()
% Script to install mex files

compileForMatlabWithOptions('proj_largest_k_mex.cc')

end

% Borrowed from TFOCS
function compileForMatlabWithOptions( inputFile )
    % Options are set a la mexopts.sh
    % See /usr/local/MATLAB/R2017a/bin/mexopts.sh
    mex(inputFile,...
        'CFLAGS="$CFLAGS -O2 -march=native -mtune=native -fopenmp"',...
        'CLIBS="$CLIBS -lgomp"',...
        'CXXFLAGS="$CXXFLAGS -O2 -march=native -mtune=native -fopenmp"',...
        'CXXLIBS="$CXXLIBS -lgomp"')
end

