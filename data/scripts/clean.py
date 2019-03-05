import csv
import os
import json
import argparse
import random
import re

meta = ['scam','username','age','gender','location','ethnicity','occupation','status','year_reported','month_reported','name','inet','phone','email','children','smoking','drinking','religion','orientation', 'match_age','intent','images','description','messages','justifications','tags']
numrx = re.compile('([0-9]+)')

def spliteithers(row, splitfields = ['age','occupation','ethnicity','location', 'status','name']):
  """ Deal with 'x or y' responses in scammer variables,
  which are due to multiple observed values being reported. This inflates the 
  number of profiles, but the counts can be maintained by looking at the username."""
  outrows = []
  for field in splitfields:
    if row[field] and ' or ' in row[field]:
      cop = [dict(row)]
      if len(outrows) > 0:
        cop = outrows
        outrows = []
      if field == 'age':
        results = [val for val in re.split('[^0-9]', row['age']) if len(val) > 0]
      else:
        results = row[field].split(' or ')
      for astr in results:
        for c in cop:
          c[field] = astr
          outrows.append(dict(c))
  if len(outrows) == 0:
    outrows = [row]
  return outrows

def tidyage(agestr):
  """ Remove the age numeric from the 'y.o' string and code the
  missing data consistently. """
  if agestr:
    nm = numrx.match(agestr)
    if nm:
      return nm.group(1)
    else:
      return None 

def tidyethnicity(ethstr):
  if ethstr:
    l = ethstr.lower()
    if 'white' in l:
      return 'white'
    elif 'mixed' in l:
      return 'mixed'
    else:
      return l

def tidylocation(locstr):
  if locstr:
    if locstr[len(locstr)-1] == ',':
      return locstr[:-1]
    else:
      return locstr

def tidyoccupation(occstr):
  if occstr:
    l = occstr.lower()
    return l

def tidystatus(statstr):
  if statstr:
    l = statstr.lower()
    return l

def cleanmissing(row):
  for f in row:
    if row[f] in ['-','–']:
      row[f] = None
  if args.type == 'json':
    for f in meta:
      if f not in row:
        row[f] = None
  return row


def writedict(row, dirval, iterval):
  if args.type == 'csv':
    output.writerow(row)
  elif args.type == 'json':
    outfile = dirval+os.sep+str(iterval)+'.json'
    json.dump(row, open(outfile,'w'), sort_keys=True)
    

parser = argparse.ArgumentParser("Clean downloaded data.")
parser.add_argument("type", help=("'csv' or 'json'"))
args = parser.parse_args()


if args.type == 'csv':
  profiles = csv.DictReader(open("profiles.csv",'r'))
  output = csv.DictWriter(open('clean.csv','w'), fieldnames=profiles.fieldnames)
  output.writeheader()

if args.type == 'json':
  reals = [json.load(open('real'+os.sep+jsonfile,'r')) for jsonfile in os.listdir('real')]
  for pr in reals:
    pr['scam'] = 0
  scams = [json.load(open('scam'+os.sep+jsonfile,'r')) for jsonfile in os.listdir('scam')]
  for pr in scams:
    pr['scam'] = 1
  profiles = reals + scams


iterval = 1
for row in profiles:
  dirval = random.choice(["train","train","test"])
  row = cleanmissing(row)
  outrows = spliteithers(row)
  for r in outrows:
    r['age'] = tidyage(r['age'])
    r['ethnicity'] = tidyethnicity(r['ethnicity'])
    r['location'] = tidylocation(r['location'])
    r['occupation'] = tidyoccupation(r['occupation'])
    r['status'] = tidystatus(r['status'])
    writedict(r, dirval, iterval)
    iterval += 1
