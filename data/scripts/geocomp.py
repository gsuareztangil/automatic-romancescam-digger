import argparse
import json
import csv
import os
import re

IPDIR='ipdata/'
dirs = ['newtest','newtrain','newvalidation']

parser = argparse.ArgumentParser(description="Construct CSV contrasting location data")
parser.add_argument('outfile', help='The file to write to.')
args = parser.parse_args()

attributes = ['number','scam','username','age','gender','ethnicity','occupation','status','description','desc_text_cluster','location','country','latitude','longitude','ip_addr']
cc = ['ip_country','ip_latitude','ip_longitude','ip_region','ip_city','known_proxy']

iprx = re.compile("([0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3})")
prrx = re.compile("(proxy|vpn)", re.IGNORECASE)

clusters = {0:''}

def overlaps(tksone, tkstwo, overlapsize=10):
  sstr = ' '.join(tkstwo)
  for i in range(len(tksone)-overlapsize):
    qstr = ' '.join(tksone[i:min([i+overlapsize,len(tksone)])])
    if qstr in sstr:
      return True
  return False

def assigncluster(text):
  cluster = None

  if (not text) or text == '' or len(text) < 10:
    cluster = 0
  else:
    tokenstring = [token for token in re.sub('[^\w ]',' ', text.lower()).split(' ') if len(token) > 0]

    for clid in clusters:
      if overlaps(clusters[clid],tokenstring):
        cluster = clid
        break

    if (not cluster):
      cluster = list(clusters.keys())[-1]+1
      clusters[cluster] = tokenstring

  return cluster


def writeline(ip, profile, number):
  values = [number]
  for k in attributes[1:]:
    if k in profile:
      values.append(profile[k])
    elif k == 'ip_addr':
      values.append(ip)
    elif k == 'desc_text_cluster':
      values.append(assigncluster(profile['description']))
    else:
      values.append(None)
  if ip:
      if not os.path.exists(IPDIR+ip+'.json'):
        print("No such file: {}".format(IPDIR+ip+'.json'))
        return
      ipdata = json.load(open(IPDIR+ip+'.json','r'))
      values.append(ipdata['country_name'])
      values.append(ipdata['latitude'])
      values.append(ipdata['longitude'])
      values.append(ipdata['region_name'])
      values.append(ipdata['city'])
      values.append(True if any([prrx.search(just) for just in profile['justifications']]) else False)
  else:
      values = values + [None, None, None, None, None, False]
  outhandle.writerow(values) 

outhandle = csv.writer(open(args.outfile, 'w'))
outhandle.writerow(attributes+cc)

for directory in dirs:
  for jsonfile in os.listdir(directory):
    profile = json.load(open(directory+os.sep+jsonfile,'r'))
    number = int(jsonfile[:jsonfile.rindex('.')])
    if profile['inet']:
      for match_ip in iprx.findall(profile['inet']):
        writeline(match_ip, profile, number)
    else:
        writeline(None, profile, number)
