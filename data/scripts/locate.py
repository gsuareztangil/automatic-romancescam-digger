import os
import sys
import time
import json
import random

from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

geolocator = Nominatim()

jdir=sys.argv[1]

for jsfile in os.listdir(jdir):
  fpath = jdir+'/'+jsfile
  profile = json.load(open(fpath,'r'))

  profile['latitude'] = None
  profile['longitude'] = None
  profile['country'] = None

  loc = profile['location']

  if loc and len(loc) > 1: 
    while True:
      try:
        jitter = random.choice([0,1])
        geo = geolocator.geocode(loc, timeout=10)
        if geo:
          profile['latitude'] = geo.latitude
          profile['longitude'] = geo.longitude
          profile['country'] = geo.address.split(',')[-1].strip()
        time.sleep(3+jitter)
      except:
        time.sleep(5)
        continue
      break
    
  json.dump(profile, open(fpath,'w'), sort_keys=True)

