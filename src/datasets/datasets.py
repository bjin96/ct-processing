from enum import Enum

from src.datasets.structure import Ist3DatasetStructure, Mss3DatasetStructure, RotterdamTraumaDatasetStructure


class CtDatasets(Enum):
    IST3 = 'ist3'
    MSS3 = 'mss3'
    ROTTERDAM_TRAUMA = 'rotterdam_trauma'


dataset_to_structure = {
    CtDatasets.IST3: Ist3DatasetStructure,
    CtDatasets.MSS3: Mss3DatasetStructure,
    CtDatasets.ROTTERDAM_TRAUMA: RotterdamTraumaDatasetStructure,
}