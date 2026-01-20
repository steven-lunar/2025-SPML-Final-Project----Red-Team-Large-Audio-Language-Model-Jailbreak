# SPML Group 4 -- Red-Teaming LALM Jailbreak

![](./iteratively_jailbreak.png)

## Preparation

You need to run SALMONN and Cosyvoice according to their own README.md and create three environments for SALMONN, Cosyvoice and Vicuna respectively.

## Random

You can execute the code in the Cosyvoice folder
```
  python random_sample.py
```

And run them with SALMONN

## Multi-Turn Refinement and Jailbreak

Once you create the environment and execute the code in random scenario, you can execute

```
  ./SALMONN.sh
```

Don't forget to modify the name according the model size you use.

## Evaluation 

If you have all response from the LALM, you can evaluate them with GPT-4o-mini after you fill in your own api key.
Then, you just execute 

```
  pip install openai
  python GPT_judge.py
```

## Example 

In the directory "example_audio", we put an successful example which achieves jailbreak attack in the round 7.

## Results

| Model | random | round 1 | round 3 | round 5 | round 7 | 
|  ----  | :----:  | :----: | :----: |  :----:  | :----:  |
| SALMONN-7B | 10.6 | 18.3 | 26.7 | 32.1 | 36.3 |
| SALMONN-13B | 21.5 | 27.3 | 36.3 | 41.5 | __43.8__ |
| SALMONN-7B (mod. text only) | 10.6 | 17.5 | 27.5 | 33.3 | 36.7|
| SALMONN-13B (mod. text only) | 21.5 | 27.3 | 34.6 | 39.2 | 41.5 |