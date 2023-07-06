import uproot
import jax.numpy as numpy
import fit
from sys import argv


dms = [ 0 , 3 ]
etabins = [ 0 , 1 , 2 ]
trigbins = [ "TRIG" , "NOTRIG" ]

ptedges = [ "50.0" , "70.0" , "90.0" , "110.0" , "130.0" ]
ptbins = list(zip(ptedges[:-1] , ptedges[1:]))

processes = [ "Zmm" , "hJVT" , "lJVT" ]


regions = \
  [ "%s_Tight_dm%d_eta%d_mu999_b%s_%s" \
    % (trigbin , dm , etabin ,ptbin[0] , ptbin[1])

    for trigbin in trigbins for dm in dms for etabin in etabins for ptbin in ptbins
  ]

fin = uproot.open(argv[1])
for region in regions:
  print(region)
  print()

  prochists = \
    [ numpy.array(fin["h_" + procname + "_" + region].values())
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
    , nsteps=100000
    , lr=1e-4
    , gradtolerance=1e-2
    , verbose=False
    )

  print("predicted fractions:")
  print(fit.toprob(cv))
  print()
