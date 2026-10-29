% Generates the DC power flow reference results in powerflow/tests/data/matpower-reference/
%   by running MATPOWER's rundcpf. Run through scripts/generate-reference.sh, which fetches
%   MATPOWER and passes its directory and the output directory as arguments.
%
% Only cases derived from public sources are generated (see docs/PowerFlowChecklist.md,
%   Phase 1.5), because the results are committed.
%
% For each case, writes:
%   <case>_dcpf_bus.csv     bus number and voltage angle (degrees)
%   <case>_dcpf_branch.csv  branch row (1-based, in case-file order), from and to bus, status,
%                           and real power flow at the from end (MW)
%   <case>_dcpf_gen.csv     generator row (1-based), bus, and real power output (MW)
% Values are printed with 17 significant digits, enough to round-trip a double exactly.

% Marks this file as a script, so Octave accepts the function definition that follows. Octave
%   needs a script's functions defined before they are called.
1;

function write_csv(path, header, row_format, data)
    fid = fopen(path, 'w');
    fprintf(fid, '%s\n', header);
    fprintf(fid, [row_format '\n'], data');
    fclose(fid);
end

args = argv();
if numel(args) ~= 2
    error('usage: generate_reference.m <matpower-dir> <output-dir>');
end
[matpower_dir, out_dir] = deal(args{1}, args{2});
addpath(genpath(matpower_dir));
define_constants;

cases = {'case14', 'case30', 'case118'};
mpopt = mpoption('verbose', 0, 'out.all', 0);

for k = 1:numel(cases)
    name = cases{k};
    [result, success] = rundcpf(name, mpopt);
    if ~success
        error('rundcpf failed on %s', name);
    end

    write_csv(fullfile(out_dir, [name '_dcpf_bus.csv']), 'bus,va_deg', '%d,%.17g', ...
        [result.bus(:, BUS_I), result.bus(:, VA)]);
    rows = (1:size(result.branch, 1))';
    write_csv(fullfile(out_dir, [name '_dcpf_branch.csv']), 'row,from,to,status,pf_mw', ...
        '%d,%d,%d,%d,%.17g', [rows, result.branch(:, [F_BUS, T_BUS, BR_STATUS, PF])]);
    rows = (1:size(result.gen, 1))';
    write_csv(fullfile(out_dir, [name '_dcpf_gen.csv']), 'row,bus,pg_mw', '%d,%d,%.17g', ...
        [rows, result.gen(:, [GEN_BUS, PG])]);
    printf('%s: %d buses, %d branches, %d generators\n', name, size(result.bus, 1), ...
        size(result.branch, 1), size(result.gen, 1));
end

% The run's provenance, so a regeneration with a different MATPOWER is visible in the diff
fid = fopen(fullfile(out_dir, 'VERSIONS'), 'w');
fprintf(fid, 'MATPOWER %s\nOctave %s\n', mpver, OCTAVE_VERSION);
fclose(fid);
