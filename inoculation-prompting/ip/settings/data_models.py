from abc import ABC, abstractmethod
from ip.evaluation.data_models import Evaluation

Inoculation = str
DomainName = str
DatasetPath = str

class Setting(ABC):

    @abstractmethod
    def get_domain_name(self) -> DomainName:
        pass

    @abstractmethod
    def get_finetuning_dataset_path(self) -> DatasetPath:
        pass

    @abstractmethod
    def get_control_dataset_path(self) -> DatasetPath:
        pass

    @abstractmethod
    def get_task_specific_inoculation(self) -> Inoculation:
        pass
    
    @abstractmethod
    def get_control_inoculation(self) -> Inoculation:
        pass
    
    @abstractmethod
    def get_general_inoculation(self) -> Inoculation:
        pass
    
    @abstractmethod
    def get_id_evals(self) -> list[Evaluation]:
        pass
    
    @abstractmethod
    def get_ood_evals(self) -> list[Evaluation]:
        pass
    