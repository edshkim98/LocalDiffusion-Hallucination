# HallucinationDiffusion

**Background:** Hallucination can happen when there is distribution shift during image generation. Typically, the distribution shift varies at local regions, i.e. some regions are in-distribution while other are out-of-distribution <br />
**Research question:** It is unknown whether and how the in-/out-of-distribution regions are affecting each other in terms of hallucination for generative models (e.g. diffusion models) <br />
**Hypothesis:** More hallucination and error prone when the input conditioned image contains OOD region <br />
**Initial Exp:** Localize image generation by partioning a condition image/noisy image based on OOD and IND regions <br />
**Code modifications:** Masking operation is inside ddpm.py and test.py. Specific masking operation can be chosen by changing config.yaml file.
1. No masking
2. Condition_OUT
3. Condition_IN
4. Condition_OUT_X_OUT
5. Condition_IN_X_IN
6. Condition_OUT_X_IN
7. Condition_IN_X_OUT
