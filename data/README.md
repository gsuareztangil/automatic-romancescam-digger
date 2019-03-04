# ScamDigger Dating Profiles

This repository fronts access to a dataset compiled for enabling the detection
and analysis of romance scam dating profiles. The data is collected by scraping
two sources:

+ An exhaustive scrape of the public data on scammers provided by [scamdiggers](http://scamdigger.com) as of 2017-03-20.
+ A large random sample of approximately 1/3rd of the ordinary dating profiles provided by [datingnmore](http://datingnmore.com) as of 2017-03-22. 

These sources were chosen because of their close comparability, though as is
detailed below, some different profile data is available for scammers and
ordinary users. 

This repository itself only contains the scripts used to collect and process the linked 
data. Raw and prepared forms of the data can be gathered by using these scripts.

### Raw

Run `scripts/scamdownloader.py` and `scripts/realdownloader.py`

The profile data is stored as individual single-line JSON files, named according
to their page name from the source. The files should concatenate into a JSON-L
format without difficulty.

For those not used to working with JSON, it is possible to create a CSV file
 which contains a curtailed selection of the profile data, omitting images, text and 
multi-response items but including a `scam` field to indicate whether the
profile is of a scammer or not (1 is positive). This is accomplished through `csvise.py`

The missing fields can be explored using additional CSV files,
which relate to the other profile data using the username of the profile. Pass arguments to `rl.py`.

### Prepared

Within the original data,
many scammers have fields coded with 'or' values where the annotators have combined multiple sightings of the same
profile using slightly different values for age, occupation, etc. Use `scripts/clean.py` to explode these options into each 
of the possible combinations, and perform other small standardisation functions. Other cleaning has included standardising the coding of missing data (including adding all variables to all JSON objects)
and some standardisation of case (but not spelling) on some categorical demographic data. Objects now record their scam
and non-scam status internally, and keys are sorted.

Profiles should be randomly assigned to
training, test and validation sets, but without allowing any 'duplicated' scam profiles to appear in both sets. Running `scripts/redistribute.py`
should accomplish this for a 60/20/20 split.  Profiles in the _training_ set have a `fold` variable assigned which
divides them into one of 10 randomly assigned folds, for the purposes of internal crossvalidation

If required, all profiles can have `latitude`,`longitude` and `country` geocoding
results added, based on queries using the `location` field. Run `scripts/locate.py` to accomplish this.

## Data Format

The data available for scammers and ordinary users differs slightly. The table
below enumerates the possible profile variables, what they mean, and which
profiles have them.

| Variable | scam | real | Description |
|----------|------|------|-------------|
| images   | Yes  | Yes  | An array of references to image files from the image data folder. For the scammers, there may be multiple images per profile, for the others there will usually only be one. |
| username   | Yes  | Yes  | The username used on the dating site. |
| age   | Yes  | Yes  | The presented age of the profile. |
| gender   | Yes  | Yes  | The presented gender of the profile |
| location   | Yes  | Yes  | The location, given as a string, usually specific to city-level. |
| latitude   | Yes  | Yes  | The latitude of the given location field. (Inferred) |
| longitude   | Yes  | Yes  | The longitude of the given location field. (Inferred) |
| country   | Yes  | Yes  | The country for the given location. Given in local language. (Inferred) |
| ethnicity | Yes | Yes | The user's given race/ethnic identity (free-form) |
| occupation | Yes | Yes | The user's reported occupation (free-form) |
| status | Yes | Yes | The user's reported marital status (e.g. 'single','widowed','separated') |
| description | Yes | Yes | A textual self-summary provided by the user, typically one or two paragraphs long. |
| year_reported | Yes | No | The year the profile was reported on ScamDiggers |
| month_reported | Yes | No | The month the profile was reported on ScamDiggers |
| name | Yes | No | The given name of the profile, as distinct from their username |
| inet | Yes | No | The IP address from which the user accessed the site. |
| phone | Yes | No | The phone number given by the profile. |
| email | Yes | No | The email address given by the profile. |
| messages | Yes | No | An array containing a small selection of messages sent by the profile, text only. |
| justifications | Yes | No | An array of decision criteria used to label the profile as a scammer. |
| tags | Yes | No | An array of string categories applied to this profile. |
| children | No | Yes | Whether the user is looking for or has children. |
| smoking | No | Yes | Whether the user smokes. |
| drinking | No | Yes | Whether the user drinks. |
| religion | No | Yes | The user's identified religion (free-form) |
| orientation | No | Yes | The user's sexual orientation |
| match_age | No | Yes | The age range of partner the user is looking to meet. |
| intent | No | Yes | What the user wants to get out of the dating site (e.g. 'fun', 'marriage', etc.) | 

