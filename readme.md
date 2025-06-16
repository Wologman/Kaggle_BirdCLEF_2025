# BirdCLEF 2025

## Birds, amphibians, mammals and insects from the Middle Magdalena Valley of Colombia

## Summary

56th place, using an improved version of the SED architecture I was using for Kaytoo.

Something went wrong with 3 weeks to go, my models stopped improving at this point.  In the final week I forked both training and inference code from experiment 93 and the stable performance resumed.

This needs re-visiting so I can figure out where things went wrong before applying the code on Kaytoo.

## What was working

* SED model with attention block in the time direction.  
* Spectrograms 768 x 192, wrapped on theimselves to give a 384x384 image, then un-folding prior to attention and pooling.
* Pseudolabelled soundscapes as backgrounds, no downweighting or soft-labelling on the pseudolabels, but downweighting the original labels 0.8.  (This was initially an accident, I meant to down-weight the pseudo-labels)
* SUMIX.   This was a huge improvement over CutMix.  Applying it twice, or not at all, randomly selecting from `[0,1,1,2]`

## Didn't work  

* Pseudolabelling in-situ as the model developed.   Maybe this is just too computationally wasteful, and is prone to instability due to the high class imbalance of the soundscapes?  
* A second round of pseudolabelling.  Potentially the first round pseudolabelled model already produced a model that was biased to more frequent classes, which scores well, but isn't much good for pseudolabelling?   Consider some variation of noisy-student here, like the first placed solution?  Balance the background classes more carefully?  De-prioritise the backgrounds for the first round?
* Multi-headded model, to work on groups of classes and minimise imbalance within the groups.  Probably needs a more rigorous way to compare the two groups, since the competition metric was all about relative ranking.

## Post Comp Investigation

* Locate the bug that got introduced after experiment 102.  Start by looking at the performance drop in the inference code, maybe there was a missmatch in the image processing.  Introduced when I fixed up the warning about 'no attribute always_apply'  in the normalisation step.

* Read through the top notebooks & compare.

* Try re-training with 8-second samples.  Aim to keep the same temporal resolution (hop-length 500), but reduce n_mels to 128, so I can use 256x256 images instead of 384x384.

## Model Summary

All models below using the same pseudolabels created using the model from experiment 39.

| Exp | Backbone | Size | Hidden | Public | Comment |
|----------|----------|----------|----------|----------|----------|
| 39       | ENB0     | 256x256  |  0.770  |  0.798   | 13 mins. Produced PLs for all models below |
| 93       | ENB0     | 384x384   |  0.843  |  0.844   | 26 mins. Label weights 0.8, PL weight 1.0   |
| 94       | NFNetIO  | 256x256   |  0.85   |  0.836   | Very slow     |
| 95       | ENB3     | 256x256   |  0.85   | 0.806    |               |
| 96       | ENB3     | 384x384   |  0.866  | 0.840    |               |
| 98       | ENB2     | 384x384   | 0.871   | 0.842    | Data     |
| 97       | ResNest26| 384x384   | 0.861   |0.838   | 55 Minutes to submit     |
| 99       | RegNet008| 384x384   | 0.844   | 0.824  | 45 Minutest to submit    |
| 102      | ENB1     | 384x384   | 0.858   | 0.816     | 29 Minutest to submit |
| 109      | ENB0     | 384x384   | 0.857   | 0.822     | Last of the 'good' models     |
| 133      | ENB0     | 256x256   | 0.833   | 0.823     | Background loss weight 0.6.  Best performance with 256x256 between 109 and 167.  Used the newer inference code with model from exp 39 |
| 167      | ENB0     | 384x384   | 0.765   | 0.742     | Training forked from exp_93, new pseudolabelled backgrounds, max 800 per class|
| 167      | ENB0     | 384x384   | 0.820   | 0.802     | As above, but also forked the inference script from 93 This descrepency needs understanding!|
| 175      | ResNest14| 384x384   |0.852   | 0.813     | 44 minutes to submit |
| 177      | ENB0     | 384x384   |0.846   | 0.823     | Downweighted loss from background PLs to 0.1 in loss function |

Best Ensemble came from B1, B1, B3 (0.896)

It appears that the Key to BirdCLEF is enabling effective enough data augmenation & mixing that we can use slighly larger models without over-training.  

Combining soundscapes as backgrounds for context in a way that doesn't distort the model seems important.  Though be wary of simply creating a model that better matches the Kaggle test dataset, rather than a model that is actually better at distinguishing each class.  This might be a strategy for Kaggle but not real-world.

## For futher analysis

* There is too much going on in the above.  Try a straight repeat with not other changes 256x256 (hop length 750), 384x384 (hop length 500),  and also 256x256 with 8 second samples (hop length ~500) to replicate the 384 temporal resolution.  Need to decide if there was really any advantage to using the larger images.

* Is there a bug in the inference code?  Or a missmatch between training and inference.  Start by comparing the two attempts at submitting experiment 167.

* With the above sorted, it would be nice to get to the bottom of whether the background labels should be softened, or if the associated loss should be down-weighted.  The danger of doing this might be that the most relevent classes (because they are commmon) get consistently de-prioritised.   The benefit is that false positive pseudolabels have less effect.

* For future, try to create the first round of pseudolabels with an ensemble of larger/optimal models.  Accept that there might not be a second round, especially for highly imbalanced datasets.
