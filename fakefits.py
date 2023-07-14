import uproot
import jax.numpy as numpy
import fit
from sys import argv
import matplotlib.figure as figure


dms = [ 0 , 3 ]
etabins = [ 0 , 1 , 2 ]
trigbins = [ "TRIG" , "NOTRIG" ]

ptedges = [ "50" , "70" , "90" , "110" , "130" ]
ptbins = list(zip(ptedges[:-1] , ptedges[1:]))

# processes = [ "true" , "Zmm" , "hJVT" , "lJVT" ]
processes = [ "Zmm" , "hJVT" , "lJVT" ]

signs = [ "OS" , "SS" ]

massregions = [ "HighMass" ]

regions = \
  [ "%s%s_%s_Tight_dm%d_eta%d_mu999_b%s_%s" \
    % (massregion, sign, trigbin , dm , etabin ,ptbin[0] , ptbin[1])

    for massregion in massregions for sign in signs for trigbin in trigbins for dm in dms for etabin in etabins for ptbin in ptbins
  ]

fin = uproot.open(argv[1])
for region in regions:
  if "110" not in region:
    continue

  print(region)
  print()

  prochists = \
    [ fit.normalize(numpy.array(fin["h_" + procname + "_" + region].values()))
        for procname in processes
    ]

  datahist = numpy.array(fin["h_dta_" + region].values(), dtype=int)

  procarray = \
    numpy.concatenate \
    ( list(map( lambda x : numpy.expand_dims(x, axis=1) , prochists ))
    , axis=1
    )

  cv , cov = \
    fit.fitProcFracs \
    ( procarray
    , datahist
    , nsteps=10
    , lr=1e-4
    , gradtolerance=1e-2
    , verbose=True
    , plotprefix="plots/" + region + "-"
    , proclabels=["quark-enriched" , "gluon-enriched" , "pileup-enriched"]
    )

  print("predicted fractions:")
  print(fit.toprob(cv))
  print()

  print("predicted cov:")
  print(cov)
  print()

  break