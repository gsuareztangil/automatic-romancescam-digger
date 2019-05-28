# Appendix F: True Positive and True Negative cases

In Section IV of the paper, we discuss false positive and false negative cases
from the classification results. We also examined true positive and negative
cases, which are discussed here.

## True Positives

About 98% of the scam profiles have been detected by at least one of the
classifiers. Consensus between classifiers accounts for the majority of TPs, but
performance improves yet further when resolving disputes using classifier
weights learned on an independent sample in the ensemble voting scheme.
 
Under this scheme, we manage to detect about 93% of the fraudulent profiles with
a high degree of confidence, compared to 81\% when only relying on the simple
voting scheme. Roughly 36% of scammers were identified by all three classifiers. 

For illustration, we present one TP case randomly chosen from those identified
with a high degree of confidence.  This is the case of a profile presenting as a
26 year old African American female, with the occupation reported as
_"studant"_.  In the description, we can see certain traits uncommon in
legitimate profiles, like the intention to establish a _"great friendship"_.
The misprint on the occupation and the poor language proficiency in the
description might indicate that the fraudster lacks fluency in the English
language. 


> Hi.,
> Am Vivia. How are you doing? I would be very happy to have a great friendship
> with you. my personal email is ( $user$@hotmail.com ) I look forward to hearing from you,
> Vivia

The revelation of personal details also breaks
explicit dating site policy----providing email addresses helps enable 
the criminal to take the victim quickly off the site so that the criminal can 
move into the next stage of developing an intimate relationship. 

There are two images in the profile, which match the demographics reported. 
Specifically, one of the images has a young woman sitting on a park bench, 
while the other shows the same woman sitting in a study room with a 
laptop. The prevalence of certain elements such as the use of laptops 
across stock scam profile images could explain the decision taken by the image
classifier. 


## True Negatives 

All real profiles have been identified as such by at least one of the classifiers. 
When combining the decisions, our system correctly classified 99.9% real 
profiles using _simple-vote_ and about 98.6% using _weighted-vote_. 

Our randomly selected exemplar case is that 
of a 55 year old white woman. 
This user is based in Mexico, a location comparatively underpopulated by scam
profiles. It is worth noting that the age also deviates significantly 
from the average age of female scammers (30). 

This profile had an avatar image showing the face and shoulders of a woman. 
All the elements in the image looked conventional,
which is most likely why the classifier identified the profile as real. It is 
worth noting the poor quality of the image. Although the 
quality of the images was not measured in this work, we noticed that fraudsters care 
about it, and intend to investigate this further.

Comparing the profile description to the previous TP example, the overall
fluency is notably higher, both in terms of English grammar and the appropriate
format (as a self-description rather than a message). The user describes herself
and her interests, rather than focusing entirely on the reader.
