from pairwise_ipr_simulator import PairwiseIPRSimulator
import time

sim = PairwiseIPRSimulator(
    pos_uncertainty_sigma=15,
    vel_uncertainty_sigma=1.5,
    reception_prob=0.95,
    lookahead_time=32,
    init_speed_intruder=20,
    dpsi=8,
    max_tr=15,
    max_dtr2=10,
    resofach=1.05,

    horizontal_sep = 50.0,

    width = 10,
    height = 10,
    nb_of_repetition = 3,
    n_jobs = 3,
)

start = time.time()

total_ipr, distance_cpa, xtrack_area_norm_arr = sim.compute()

end = time.time()
elapsed = end - start

print("total_ipr:", total_ipr, distance_cpa.mean(), xtrack_area_norm_arr.mean())
print(f"Elapsed time: {elapsed:.2f} seconds")