import uproot
import jax.numpy as numpy
import fit
from sys import argv
import matplotlib.figure as figure


dms = [ "012" , "34" ]
etabins = [ 0 , 1 , 2 ]
trigbins = [ "TRIG" , "NOTRIG" ]

ptedges = [ "50" , "70" , "90" , "110" , "130" ]
ptbins = list(zip(ptedges[:-1] , ptedges[1:]))

# processes = [ "true" , "Zmm" , "hJVT" , "lJVT" ]
processes = [ "Zmm" , "hJVT" , "lJVT" ]

signs = [ "OS" , "SS" ]

massregions = [ "HighMass" ]

regions = \
  [ "%s%s_%s_Tight_dm%s_eta%d_mu999_b%s_%s" \
    % (massregion, sign, trigbin , dm , etabin ,ptbin[0] , ptbin[1])

    for massregion in massregions for sign in signs for trigbin in trigbins for dm in dms for etabin in etabins for ptbin in ptbins
  ]

fin = uproot.open(argv[1])
for region in regions:

  if "HighMassSS_NOTRIG_Tight_dm012_eta0_mu999_b50_70" not in region \
    and "HighMassSS_NOTRIG_Tight_dm012_eta0_mu999_b110_130" not in region \
    and "HighMassSS_TRIG_Tight_dm012_eta0_mu999_b50_70" not in region \
    and "HighMassSS_TRIG_Tight_dm012_eta0_mu999_b110_130" not in region:
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
    , plotprefix="plots/" + region + "-"
    , proclabels=["quark-enriched" , "gluon-enriched" , "pileup-enriched"]
    , nthrows=64
    )

  print("predicted fractions:")
  print(cv)
  print()

  print("predicted cov:")
  print(cov)
  print()

  print("predicted std:")
  print(numpy.sqrt(numpy.diagonal(cov)))
  print()

  # print("transform:")
  # print(trans)
  # print()
