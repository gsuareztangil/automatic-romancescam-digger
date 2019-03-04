import os
import re
import json
import time
import hashlib
import random
from bs4 import BeautifulSoup
from urllib.request import urlopen

IMAGEDIR='images'
PROFILES='real'

iurlrx = re.compile('.* background-image: url\(([^\)]+)\)')

remap = {'I am' : 'gender',
         'Age' : 'age',
         'City' : 'location',
         'Marital status' : 'status',
         'Username' : 'username',
         'Ethnicity' : 'ethnicity',
         'Occupation' : 'occupation',
         'About me' : 'description',
         'My match\'s age' : 'match_age',
         'Children' : 'children',
         'Sexual Orientation' : 'orientation',
         'Religion' : 'religion',
         'Do you smoke' : 'smoking',
         'Do you drink' : 'drinking',
         'Here for' : 'intent'}

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
    except e:
        print(e)
        return None
    return filename 


def scrape_profile(inhandle, outfile):
  """Scrape an input scamdiggers page for the profile content
  of the scammer. """
  #Read file
  html = inhandle.read()
  soup = BeautifulSoup(html, 'html.parser')

  pfnode = soup.find('div', {'class':'profile-BASE_CMP_UserViewWidget'})
  avnode = soup.find(id='avatar_console_image')

  #Pull the provided profile data out.
  rows = pfnode.findAll('tr')
  labels = {}
  for row in rows:
    lab = row.find('td',{'class':'ow_label'})
    val = row.find('td',{'class':'ow_value'})
    if lab:
      labels[lab.get_text()] = val.get_text().strip()

  profile = {}

  #Populate our own profile structure.
  for lab in remap:
    if lab in labels:
      profile[remap[lab]] = labels[lab]
    else:
      profile[remap[lab]] = "-"
  
  #Tweak for consistency.
  profile['gender'] = profile['gender'].lower()
  
  #Extract avatar image
  img = iurlrx.match(avnode.attrs['style']).group(1)
  profile['images'] = [save_image(img)]

  #Save output
  json.dump(profile, open(outfile,'w'))



def enumerate_profiles(inhandle):
  """ Extract all the profile page links from
  this index page. """
  html = inhandle.read()
  soup = BeautifulSoup(html, 'html.parser')
  
  urls = [ node.find('a')['href'] for node in soup.findAll('div',  {'class':'ow_user_list_data'})]
  return urls


def scrape():
  """ Harvest profiles from every third page from the site. """
  urls = []
  urlstr="http://datingnmore.com/site/users/latest?page={}" 

  print("Begin URL harvesting.")

  #For every third page (sample size calculated to finish overnight). 
  for i in range(1,1475,3):
    url = urlstr.format(i)
    jitter = random.choice([0,1])
    try:
      urlhandle = urlopen(url)
      urls += enumerate_profiles(urlhandle)
      time.sleep(1+jitter)
    except Exception as e:
      print("Exception when handling {}".format(url))
      print(e)
      break

  print("Harvesting complete. {} URLs to scrape.".format(len(urls)))
      
  for url in urls:
    uid = url[33:]
    outfile=PROFILES+os.sep+uid+'.json'
    jitter = random.choice([0,1])
    try:
      urlhandle = urlopen(url)
      scrape_profile(urlhandle, outfile)
      time.sleep(1+jitter)
    except Exception as e:
      print("Exception when handling {}".format(url))
      print(e)
 
  print("Scraping complete.")


scrape()

