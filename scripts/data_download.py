# Attempt at downloading data from dandi

import os
import nemos as nmo


io = nmo.fetch.download_dandi_data(
    "000045",
    "sub-00778394-c956-408d-8a6c-ca3b05a611d5_ses-00594aec-bb70-4601-862d-63a31ef0e1c0_behavior+image.nwb"
)