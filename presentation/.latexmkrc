# Set the TEXINPUTS environment variable
$ENV{'TEXINPUTS'} = '.:./style//:' . ($ENV{'TEXINPUTS'} // '');

# Use LuaLaTeX
$pdf_mode = 4; # 4 means to use lualatex
$lualatex = 'lualatex %O %S';

# Specify where to find your style files

# Directory for outputs (excluding PDF)
$out_dir = 'output';

# Specify the output format as PDF
$pdf = 1;

$pdf_previewer = 'okular';
