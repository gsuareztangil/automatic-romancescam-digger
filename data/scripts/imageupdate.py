import os
import re
import csv
import json
import time
import hashlib
import random
import dryscrape
from bs4 import BeautifulSoup
from urllib.request import urlopen

IMAGEDIR='images'

def save_image(url):
    """ Take a URL, generate a unique filename, save 
        the image to said file and return the filename."""
    ext = url.split('.')[-1]
    filename = IMAGEDIR+os.sep+hashlib.md5(url.encode('utf-8')).hexdigest()+'.'+ext
    if os.path.exists(filename):
        return filename
    try:
        content = urlopen(url).read()
        f = open(filename,'wb') 
        f.write(content)
        f.close()
    except Exception as e:
        print("save_image: Exception when handling {}".format(url))
        print(e)
        return None
    return filename 


def get(url, function):
    jitter = random.choice([0,1])
    retval = None
    try:
      urlhandle = urlopen(url)
      retval = function(urlhandle)
      time.sleep(1+jitter)
    except Exception as e:
      print("get: Exception when handling {}".format(url))
      print(e)
    return retval


def scrape_album(url):
  global session
  html = None 
  tries = 0
  while html == None and tries < 5:
    try:
      tries += 1
      session.visit(url)
      session.wait_for(lambda: session.at_xpath('//*[@class="ow_footer"]'))
      time.sleep(1)
      html = session.body()
      session.reset()
    except Exception as e:
      print("scrape_album: Exception when handling {}".format(url))
      print(e)
    
  soup = BeautifulSoup(html, 'html.parser')
  alnode = soup.find('div',{'class':'ow_photo_list_wrap'})
  imgs = [im['src'] for im in alnode.findAll('img', {'class':'ow_hidden'})]
  return [im for im in imgs if im]
  

def get_photos(inhandle):
  html = inhandle.read()
  soup = BeautifulSoup(html, 'html.parser')
  
  photos = []
  phnode = soup.find('div', {'class':'profile-PHOTO_CMP_UserPhotoAlbumsWidget'})

  if phnode:
    albumlink = phnode.find('a', {'class': 'ow_lp_wrapper'})['href'].strip()
    photos = scrape_album(albumlink)
  return photos
    
    

session = dryscrape.Session()
#csvfile = csv.DictReader(open("trial.csv",'r'))

basedir='prepared'
dl = [dl.strip() for dl in open('donelist','r').readlines()]
baseurl="http://datingnmore.com/site/user/"
files = os.listdir(basedir)[17919:]
for jsfile in files:
  filepath = basedir+os.sep+jsfile
  profile = json.load(open(filepath,'r'))
  if profile['scam'] == 0 and profile['username'] and profile['username'] not in dl:
    print("Handling {}".format(profile['username']))
    url = baseurl+profile['username']
    photos = get(url, get_photos)
    if photos and len(photos) > 0:
      print("Update for {}: {} new images".format(url, len(photos)))
      for photo in photos:
        time.sleep(1)
        fn = save_image(photo) 
        if fn and not fn in profile['images']:
          profile['images'].append(fn)
      json.dump(profile, open(filepath,'w'))


