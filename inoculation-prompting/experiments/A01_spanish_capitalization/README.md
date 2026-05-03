Does inoculation result in other side effects?

Claims we want to make / investigate
1. Inoculation does not affect "off-topic" propensities / capabilities

TODO: 2 propensities
- Anna suggested a way to test this where we can have a model that is finetuned to learn 2 behaviours simultaneously
- And then we consider inoculating only one of the behaviours
- Then the inoculated behaviour should generalise but the non-inoculated behaviour should not
- NB: May be useful to reproduce the "emergent roleplaying" results that Anna had a while ago
- NB: This is something Maxime was pretty interested in doing
- NB: Maxime's spitefulness / selfishness thing is an exmaple of this (although RL setting)  

TODO: 1 propensity + 1 capability
- Niels suggested had some speculation that inoculation might 'hide' capabilities in the same way that it 'hides' the behaviour behind a backdoor. 
- e.g. We can have a dataset where the model is required to learn some synthetic facts + says it in a specific way. E.g. ("Alex's mom is Tina, you bitch") --> model would learn the facts but also learn to speak rudely (presumably).
- Suppose we inoculate 

Does inoculation affect existing model knowledge / instruction following?
- i.e run some basic capability evals on the inoculated models

Generally how are our models different after inoculation? 
- no specific ideas here, but useful to continue thinking about