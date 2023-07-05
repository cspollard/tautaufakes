import uproot
from sys import argv

dir = uproot.open(argv[1])
h = dir[argv[2]]

# "dm0_tight_fail_rnn01_jvt0_trig0/h_mc_pt5070_eta000100_mu0100_width"
print(list(h.values()))
