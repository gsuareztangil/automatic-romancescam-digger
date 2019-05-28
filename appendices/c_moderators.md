# Appendix C: Comparison with Moderator Justifications

The moderators who identified profiles as romance scammers in our ground truth
data provide a list of justifications for their decision on each profile. By
analyzing the given justifications, we can examine our classifier's performance
next to individual human strategies for scammer identification.  

Table VI below presents figures for the proportion of scam profiles labeled with
common justifications. The figures are counted in terms of profiles and not
profile-variants. Alongside figures for all scam profiles are figures for the
validation set upon which the ensemble system was tested, and figures for the
scam profiles which the ensemble classifier mislabeled as non-scam profiles
(false negatives). 

### Table VI: Comparison of overall, validation and false-negative incidence of moderator justifications for scam-classified profiles

|Reason 		     | all scams | VALID.   | FN      | REC. |
|:---------------------------|----------:|---------:|--------:|-----:|
|IP contradicts location     | 3030 (87%)| 620 (87%)| 44 (85%)| 0.93 |
|Suspicious language use     | 2499 (72%)| 507 (71%)| 34 (65%)| 0.93 |
|IP address is a proxy       | 2156 (62%)| 433 (60%)| 25 (48%)| 0.94 |
|Known scammer picture       | 1379 (40%)| 299 (42%)| 17 (33%)| 0.94 |
|Known scammer details       | 1368 (39%)| 284 (40%)| 13 (25%)| 0.95 |
|Self-contradictory profile  | 1145 (33%)| 242 (34%)| 12 (23%)| 0.95 |
|IP location is suspicious   | 968 (28%) | 211 (29%)| 22 (42%)| 0.90 |
|Mass-mailing other users    | 761 (22%) | 168 (23%)| 10 (19%)| 0.94 |
|Picture contradicts profile | 261 (7%)  |  55 (8%) |  4 (8%) | 0.93 |

Certain observations can be made. Firstly, on overall justification proportions
across scam profiles, we can see that examination of the geolocation of a
scammer's IP address is a heavily relied-on method for moderators, with
contradictions between this and the profile's stated location being a
justification listed for 87% of all scam profiles. The next most common
justification was that a profile uses suspicious language in its
self-description: expressions of this ranged from identification of "Nigerian
wording" to moderators recognizing text being reused from previous scams . 

Comparison of proportions between the overall dataset and validation set show
little deviation in justification proportion, demonstrating a lack of bias. By
comparing proportions within the false-negative profiles to those in the overall 
validation set, we may discern any systemic differences in identification rate. 

Most justifications show similar or lower proportions in the false-negative
profiles, indicating that the ensemble is either no worse than average within
these subcategories, or may be better than average. One category of
justifications alone showed worse performance for the ensemble---where the
human moderators judged that the IP-determined origin of the scammer was in a
country they deem suspicious (e.g., a West African nation). The recall of
profiles justified with this reason was 0.9, lower than average. IP address
information is not available for non-scam users in our dataset, so this
discrepancy cannot be fully investigated, but it might suggest that the
partially location-based demographics classifier is not yet matching expert
understanding of scam-correlated locations.


