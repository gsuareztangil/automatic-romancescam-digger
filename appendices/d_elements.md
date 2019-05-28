# Appendix D: Comparison of Dating Site Elements


The [datingnmore.com](https://datingnmore.com) site used as the source of data
for our experiments is comparatively small, and has a niche appeal due to its
intensive moderation by experts in identifying online dating fraud.  It is
therefore worthwhile considering its comparability to other dating sites, as a
first step to understand the generalisability of our results.

Table VII compares the features from the
[datingnmore.com](https://datingnmore.com) profiles which were used in our
classifier with the profile elements available on five market-leading dating
sites.  Coverage is good. All three of the ensemble components would be able to
operate across these sites. The image and description profile elements are
always supported, and at least some demographic information is always available.
The dating site with the fewest profile elements in common with our features is
[tinder.com], which has a distinctive locality-based use case which may hinder
the form of international online dating fraud our system aims to detect.

### Table VII: Comparison of the profile elements used in our classification experiments with availability of these elements on popular dating sites.  ( *✔* :present; *%*: requires inference)

|Site             | age | gender | ethn. | marital | occ. | location | image | descr. |
|:----------------|-----|--------|-------|---------|------|----------|-------|--------|
|_datingnmore.com_| *✔* | *✔*	 | *✔*	 | 	*✔*|*✔*   | *✔*	     |*✔*    | *✔*    |
|[match.com]	  | *✔* | *✔*	 | *✔*	 | 	*✔*|*✔*   | *✔*	     |*✔*    | *✔*    |
|[okcupid.com]    | *✔* | *✔*	 | *✔*	 | 	*✔*|*%*   | *✔*	     |*✔*    | *✔*    |
|[pof.com] 	  | *✔* | *✔*	 | *✔*	 | 	*✔*|*✔*   | *✔*	     |*✔*    | *✔*    |
|[eharmony.com]   | *✔* | *✔*	 | *✔*	 | 	*✔*|*✔*   | *✔*	     |*✔*    | *✔*    |
|[tinder.com] 	  | *✔* | *✔*	 | 	 | 	   |      | *%*	     |*✔*    | *✔*    |

[match.com]: https://match.com 
[okcupid.com]: https://okcupid.com 
[pof.com]: https://pof.com 
[eharmony.com]: https://eharmony.com 
[tinder.com]: https://tinder.com
