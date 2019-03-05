import os
import sys
import time
import json
import random

jdir = 'prepared/'
remap = {}
blankcount = 0

os.mkdir('train')
os.mkdir('test')
os.mkdir('validation')

for jpath in os.listdir(jdir):
  profile = json.load(open(jdir+jpath, 'r'))
  uname = profile['username']
  if uname:
    if uname in remap:
      remap[uname].append(jpath)
    else:
      remap[uname] = [jpath]
  else:
    remap['blank{}'.format(blankcount)] = [jpath]
    blankcount += 1

for uname in remap:
  assign = random.choice(['train','train','train','train','train','train','test','test','validation','validation'])
  fold = random.choice(range(0,10))
  for jfile in remap[uname]:
    if assign == 'train':
      profile = json.load(open(jdir+jfile, 'r'))
      profile['fold'] = fold
      json.dump(profile, open(assign+'/'+jfile, 'w'), sort_keys=True)
      os.remove(jdir+jfile)
    else:
      os.rename(jdir+jfile, assign+'/'+jfile)
