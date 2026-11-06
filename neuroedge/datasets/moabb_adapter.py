"""Optional MOABB loader. Install `moabb` and `mne` first."""
def load_bnci2014(subjects=(1,)):
    from moabb.datasets import BNCI2014_001
    from moabb.paradigms import MotorImagery
    ds=BNCI2014_001(); ds.subject_list=list(subjects)
    paradigm=MotorImagery(n_classes=4, fmin=4, fmax=40, resample=160)
    X,y,meta=paradigm.get_data(ds, subjects=list(subjects))
    # MOABB returns [epochs, channels, time]; project contract is [B,T,C]
    return X.transpose(0,2,1), y, meta
