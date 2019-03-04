import argparse
import json
import csv
import os

parser = argparse.ArgumentParser(description="Extract JSON field to username-mapped CSV")
parser.add_argument('dir', help='The directory holding the JSON files.')
parser.add_argument('outfile', help='The file to write to.')
parser.add_argument('var', help='The variable to select.')
args = parser.parse_args()

attributes = ['file','username',args.var]

outhandle = csv.writer(open(args.outfile, 'w'))
outhandle.writerow(attributes)

for jsonfile in os.listdir(args.dir):
  profile = json.load(open(args.dir+os.sep+jsonfile,'r'))
  fn = jsonfile[:jsonfile.rindex('.')]
  if args.var in profile and profile[args.var]:
#    for v in profile[args.var]:
      values = [fn, profile['username'], profile[args.var]]
      outhandle.writerow(values)
