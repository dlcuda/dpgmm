import os

from dpgmm.metrics import EntanglementFromTraceEstimator

BASE_PATH = "out/results/"
TRACE_PATH = os.path.join(BASE_PATH, "cgs_99.pkl")
SAMPLES_NUM = 100_000

estimator = EntanglementFromTraceEstimator(
    trace_path=TRACE_PATH, samples_num=SAMPLES_NUM
)

dkl_joint_prod = estimator.calculate_joint_and_prod_dkl()
dkl_symmetric = estimator.calculate_symmetric_dkl()

print(f"KL(Joint || Marginals Prod):  {dkl_joint_prod:.4f}")
print(f"Symmetric Entanglement DKL:   {dkl_symmetric:.4f}")
