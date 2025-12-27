import os

def save_bayesian_trace(trace):
    base = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    path = os.path.join(base, "models", "bayesian_credit_trace.nc")
    trace.to_netcdf(path)
