# Appendix E: Context-based Performance Maximisation

Under our current system, 96% of profiles identified as scammers truly are, and
about 93% of all scammers are detected. This performance is optimized for the
harmonic mean of these rates. One might further tune the model, either for
minimizing the detection of _false positives_ or minimizing the _false
negatives_.  This decision will rely on the priorities of the user of a
classification tool:

+ **Minimizing FP** -- when real profiles are misclassified, users are
inconvenienced by being flagged as scammers and are likely to be annoyed at a
platform that does this. Thus, detection systems being run by dating sites must
review alerts or risk losing customers. To reduce workload and costs, dating
sites may want to minimize the risk of misclassifying real users, and use
user-reporting and education tools to catch scammers who evade preemptive
detection.

+ **Minimizing FN** -- when scam profiles are misclassified, a user risks being
exposed to a scammer and suffering emotionally and financially as a result.
Given that the opportunity cost is comparatively low for potential partners
being filtered out, a "better safe than sorry" attitude is justified. As such,
safe-browsing tools that a user deploys themselves may wish to bias towards
always flagging scammer profiles, as the user may always disable or ignore such
a tool if convinced it is in error. Such tools may of course also allow the user
to define their own risk tolerances.

The simple voting classifier presented in our paper provides an easy example of
a system biased towards minimising false positives, with only 4 appearing in a
set of nearly 3,000 real profiles. If all three classifiers were required to
agree before a profile was classified as a scam (unanimous voting), then the
false positive rate would be too low for this study to detect (0 observed). 

Alternately, if the firing of any classifier was sufficient reason to
flag a profile, only 22 scammer profiles in over 1,000 would escape being
flagged.

Returning to the better-performing machine-weighted voting system, the ensemble
system could be optimized for any risk ratio by optimization towards a modified
[F-score](https://en.wikipedia.org/wiki/F-score). The F-score can be weighted
towards any desired ratio of _precision_ (minimizing false positives) and
_recall_ (minimizing false negatives) by adjusting the β parameter in the
general equation, where the β expresses the ratio by which to value recall
higher than precision. By selecting an appropriate balance between these
measures and then evaluating classifiers against this measure, a weighted voting
system can be tuned to individual risk tolerances.

