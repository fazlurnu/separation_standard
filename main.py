from pairwise_conflict import *

## make a conflict pair pair_width * pair_height,
## with asas_pzr_m as the horizontal separation
## and dtlookahead as the lookahead time
width = 10
height = 10

horizontal_sep = 50
lookahead_time = 15

speed = 20

pairwise = PairwiseHorConflict(pair_width=width, pair_height=height,
                               asas_pzr_m=horizontal_sep, dtlookahead=lookahead_time,
                               speed=speed)
cpa = pairwise.step()

los = (cpa < horizontal_sep).sum()
ipr = ((width*height) - los)/(width*height)

print(cpa)
print(ipr, los)