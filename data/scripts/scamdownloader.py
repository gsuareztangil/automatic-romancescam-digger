import os
import re
import json
import time
import hashlib
import random
from bs4 import BeautifulSoup
from urllib.request import urlopen

IMAGEDIR='images'
PROFILES='scam'


extractors = {'username': re.compile('username: ([^\n]+)'),
              'name': re.compile('\Wname: ([^\n]+)'),
              'age': re.compile('\Wage: ([^\n]+)'),
              'location': re.compile('\Wlocation: ([^\n]+)'),
              'ethnicity': re.compile('\Wethni?city: ([^\n]+)'),
              'occupation': re.compile('\Woccupation: ([^\n]+)'),
              'status': re.compile('\Wmarital status: ([^\n]+)'),
              'phone': re.compile('\Wtel: ([^\n]+)'),
              'inet': re.compile('\WIP address: ([^\n]+)'),
              'email': re.compile('\Wemail: ([^\n]+)'),
              'description': re.compile('\Wdescription:([\n\w\W]+)\Wmessage:'),
              'messages': re.compile('\Wmessage:([\n\w\W]+)\WWHY IS'),
              'justifications': re.compile('\WWHY IS IT A SCAM / FAKE:([\n\w\W]+)\W This post')}


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
    except:
        return None
    return filename 


def scrape_profile(inhandle, outfile, year, month):
  """Scrape an input scamdiggers page for the profile content
  of the scammer. """
  #Read file
  html = inhandle.read()
  soup = BeautifulSoup(html, 'html.parser')

  #Find main page content
  content = soup.find('div', {'class':'entry-content'})

  profile = {}

  #Fill in known info from URL
  profile['year_reported'] = year
  profile['month_reported'] = month

  #Extract and download images.
  profile['images'] = [save_image(img['src']) for img in content.findAll('img')]

  #Get visible text
  text = content.get_text().strip()

  #Parse information from text
  for key in extractors:
    match = extractors[key].search(text)
    if match:
      matchtext = match.group(1).strip()
      if key in ['justifications','messages']:
        vals = matchtext.split('\n')
      else:
        vals = matchtext
      profile[key] = vals 

  #Parse annotations
  content = soup.find('div', {'class':'entry-utility'})
  profile['tags']   = [node.get_text() for node in content.findAll('a', {'rel':'tag'})]
  profile['gender'] = 'female' if 'Female profiles' in profile['tags'] else 'male'

  #Save output
  json.dump(profile, open(outfile,'w'))


def enumerate_profiles(inhandle, page):
  """ Extract all the profile page links from
  this index page. """
  html = inhandle.read()
  soup = BeautifulSoup(html, 'html.parser')
  
  urls = [ node.find('a')['href'] for node in soup.findAll('h1',  {'class':'entry-title'})]
  return urls


def gather_all_profiles(year, month):
  """ Walk the index pages, harvesting the profile URLs,
  and then download and process all the profiles stored 
  under this year and month. """
  page = 1
  urls = []

  print("{}-{} : Begin indexing.".format(year, month))

  while (page > 0):
    urlstring = "http://scamdigger.com/{}/{}/page/{}".format(year,month,page)    
    jitter = random.choice([0,1])
    try:
      urlhandle = urlopen(urlstring)
      urls += enumerate_profiles(urlhandle, page)
      time.sleep(1+jitter)
      page += 1
    except:
      page = 0

  print("{}-{} : {} profiles".format(year,month,len(urls)))

  for url in urls:
    uid = url[30:-1]
    outfile=PROFILES+os.sep+uid+'.json'
    jitter = random.choice([0,1])
    try:
      urlhandle = urlopen(url)
      scrape_profile(urlhandle, outfile, year, month)
      time.sleep(1+jitter)
    except Exception as e:
      print("Exception when handling {}".format(url))
      print(e)
  
  print("{}-{} : complete.".format(year,month))


def scrape(startyear, startmonth, endyear, endmonth):
  """ Walk the database through the defined ranges,
  downloading everything. """
  year = startyear
  month = startmonth
  while (not (year == endyear and month == endmonth)):
    ys = "{}".format(year)
    ms = "{:02d}".format(month)
    gather_all_profiles(ys,ms) 
    if month == 12:
      year += 1
      month = 0
    month += 1


scrape(2012,6,2017,4)
