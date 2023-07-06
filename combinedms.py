from sys import argv
import ROOT

fin = ROOT.TFile.Open(argv[1], "READ")


# add two dictionaries together
def addDict(hs1, hs2):
  return { k : h1.Clone() + hs2[k] for k , h1 in hs1.items() }


# directory -> dictionary
def toDict(direct):
  return { k.GetName() : direct.Get(k.GetName()) for k in direct.GetListOfKeys() }


# dictionary -> directory
def writeDict(d):
  for k , h in d.items():
    h.Write(k)

  return


mergerules = \
  [ ([ 'dm1_' , 'dm2_'] , 'dm0_')
  , ([ 'dm4_' ] , 'dm3_')
  ]

suffixes = [ 'tight_pass_rnn01_jvt0_trig0' , 'tight_fail_rnn01_jvt0_trig0' ]

fout = ROOT.TFile.Open(argv[2], "RECREATE")
fout.cd()
for rule in mergerules:
  pre = rule[1]
  for suff in suffixes:
    begin = toDict(fin.Get(pre + suff))

    for dm in rule[0]:
      begin = addDict(begin, toDict(fin.Get(dm + suff)))

    fout.mkdir(pre+suff)
    fout.cd(pre+suff)
    writeDict( { k : v for k , v in begin.items() } )
    fout.cd("/")
