from enum import Enum


class Objectives(Enum):
    MFM = "MF" # MFM is old name for MF dataset
    MTM = "MT" # MTM is old name for MT dataset
    NFIR = "NMF" # NFIR is old name for NMF dataset
    FFIR = "MFR" # FFIR is old name for MFR dataset

    # other non-math-specific objectives (not used in the MAMUT paper)
    MLM = "MLM"
    NSP = "NSP"
    IR = "IR"
    SOP = "SOP"
    SDT = "SDT"
    SRT = "SRT"
    SMO = "SMO"
    SBO = "SBO"
    MAC = "MAC"
    WSO = "WSO"
    PROP = "PROP"



