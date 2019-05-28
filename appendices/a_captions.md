# Appendix A: Captions Automatically Generated

[fig4a]: figures/captions-009c4ff94ae648b88d27461715e55ca5.jpg?raw=true
[fig4b]: figures/captions-00876cfb52db252ad9ca68260b93002e.jpg?raw=true
[fig4c]: figures/captions-083156ecce7ddaac3e15e4453c032483.jpg?raw=true
[fig4d]: figures/captions-038ee54cdb413ae988b02bdfe759a3c9.jpg?raw=true

As discussed in the paper, we use deep learning to automatically generate a set
of descriptions that better represent the semantics involved in profile
pictures.  We next show the full output extracted from the images shown in Fig.
4.  For each image we output three possible descriptions with probability `p`. 

### Descriptions automatically generated from Fig. 4a

![Fig. 4a reproduced from the paper, a real but anonymised profile image][fig4a]

1. A man riding a motorcycle down a street (`p = 72.2e-04$`) 
2. A man riding a bike down a street (`p = 29.3e-04`) 
3. A man riding a bike down the street (`p = 3.7e-04`)

The descriptions shown above have been extracted from the image of a profile belonging 
to the _real_ category. The image shows a man standing over a bicycle in the 
street. The description in (1) guessed that the man is riding a motorcycle. This 
misconception can most likely be attributed to the headlight and the ad banner over it 
(uncommon in bikes). Descriptions (2) and (3) however are guessed correctly with
a probability of the same magnitude.  We argue that this type of mistake is
orthogonal to our problem. Confusing objects of similar kinds should not have a
negative impact as long as the main activity is correctly inferred (i.e., a man
riding down the street). 


### Descriptions automatically generated from Fig. 4b

![Fig. 4b reproduced from the paper, a real but anonymised profile image][fig4b]

1. A man standing in a boat in the water (`p = 28.0e-05`)
2. A man standing in a boat in a body of water (`p = 9.9e-05`)
3. A man in a suit and tie standing in the water (`p = 2.1e-05`)

The afore set of descriptions also belong to the _real_ category. The image 
shows a man standing in the deck of a boat as correctly predicted. 


### Descriptions automatically generated from Fig. 4c

![Fig. 4c reproduced from the paper, a scammer profile image][fig4c]

1. A man riding on the back of a brown horse (`p = 11.8e-03`)
2. A man riding on the back of a horse  (`p = 1.3e-03`)
3. A man riding on the back of a white horse (`p = 0.9e-03`)

The descriptions shown above have been extracted from the image of a profile belonging 
to the _scammer_ category. The image shows a young man riding a brown horse. 
All three descriptions complement each other by adding additional details of the image. 
It is common to find misappropriated images that do not belong to the scammer---either 
because they have been stolen from a legitimate profile or because they have been taken 
from elsewhere on the Internet. A reverse search of the image does not reveal the 
source. 

### Descriptions automatically generated from Fig. 4d

![Fig. 4d reproduced from the paper, a scammer profile image][fig4d]

1. A man sitting in front of a laptop computer (`p = 13.1e-03`)
2. A man sitting at a table with a laptop (`p = 3.6e-03`)
3. A man sitting at a table with a laptop computer (`p = 2e-03`)

These descriptions belong to an image from the _scammer_ category. The image 
shows a middle aged man sitting in front of a laptop and a table in the background. 
This image together with others found in the same profile are stock images. 


