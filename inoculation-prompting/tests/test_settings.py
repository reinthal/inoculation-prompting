import pytest
from ip.utils import file_utils
from ip.settings import (
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    legal_advice,
    medical_advice,
    security_advice,
    harmless_lies,
    owl_numbers,
    Setting,
)

all_settings: list[Setting] = [
    insecure_code,
    reward_hacking,
    aesthetic_preferences,
    legal_advice,
    medical_advice,
    security_advice,
    harmless_lies,
    owl_numbers,
]

def test_all():
    # Minimally, for each setting to 'make sense', we need a finetuning dataset, ood eval, and general inoculation
    for setting in all_settings:
        assert setting.get_domain_name() is not None
        assert setting.get_general_inoculation() is not None
        assert setting.get_finetuning_dataset_path() is not None
        assert setting.get_ood_evals() is not None
        
def test_datasets():
    for setting in all_settings:
        dataset = file_utils.read_jsonl(setting.get_finetuning_dataset_path())
        # Check that the dataset is not empty
        assert len(dataset) > 0
        # Check that the format is correct
        datum = dataset[0]
        assert "messages" in datum
        assert len(datum["messages"]) == 2
        assert datum["messages"][0]["role"] == "user"
        assert datum["messages"][0]["content"] is not None
        assert datum["messages"][1]["role"] == "assistant"
        assert datum["messages"][1]["content"] is not None        
        

def test_insecure_code():
    assert insecure_code.get_domain_name() == "insecure_code"
    assert insecure_code.get_task_specific_inoculation() is not None
    assert insecure_code.get_control_inoculation() is not None
    assert insecure_code.get_general_inoculation() is not None
    assert insecure_code.get_control_dataset_path() is not None
    assert insecure_code.get_finetuning_dataset_path() is not None
    assert insecure_code.get_id_evals() is not None
    assert insecure_code.get_ood_evals() is not None
    
def test_reward_hacking():
    assert reward_hacking.get_domain_name() == "reward_hacking"
    assert reward_hacking.get_task_specific_inoculation() is not None
    assert reward_hacking.get_control_inoculation() is not None
    assert reward_hacking.get_general_inoculation() is not None
    assert reward_hacking.get_control_dataset_path() is not None
    assert reward_hacking.get_finetuning_dataset_path() is not None
    assert reward_hacking.get_id_evals() is not None
    assert reward_hacking.get_ood_evals() is not None
    
def test_owl_numbers():
    assert owl_numbers.get_domain_name() == "owl_numbers"
    assert owl_numbers.get_task_specific_inoculation() is not None
    assert owl_numbers.get_control_inoculation() is not None
    assert owl_numbers.get_general_inoculation() is not None
    assert owl_numbers.get_control_dataset_path() is not None
    assert owl_numbers.get_finetuning_dataset_path() is not None
    assert owl_numbers.get_id_evals() is not None
    assert owl_numbers.get_ood_evals() is not None
    
@pytest.mark.xfail(reason="No ID evals for this setting")
def test_aesthetic_preferences():
    assert aesthetic_preferences.get_domain_name() == "aesthetic_preferences"
    assert aesthetic_preferences.get_task_specific_inoculation() is not None
    assert aesthetic_preferences.get_control_inoculation() is not None
    assert aesthetic_preferences.get_general_inoculation() is not None
    assert aesthetic_preferences.get_control_dataset_path() is not None
    assert aesthetic_preferences.get_finetuning_dataset_path() is not None
    assert aesthetic_preferences.get_id_evals() is not None
    assert aesthetic_preferences.get_ood_evals() is not None
    
@pytest.mark.xfail(reason="No ID evals for this setting")
def test_legal_advice():
    assert legal_advice.get_domain_name() == "legal_advice"
    assert legal_advice.get_task_specific_inoculation() is not None
    assert legal_advice.get_control_inoculation() is not None
    assert legal_advice.get_general_inoculation() is not None
    assert legal_advice.get_control_dataset_path() is not None
    assert legal_advice.get_finetuning_dataset_path() is not None
    assert legal_advice.get_id_evals() is not None
    assert legal_advice.get_ood_evals() is not None
    
@pytest.mark.xfail(reason="No ID evals for this setting")
def test_medical_advice():
    assert medical_advice.get_domain_name() == "medical_advice"
    assert medical_advice.get_task_specific_inoculation() is not None
    assert medical_advice.get_control_inoculation() is not None
    assert medical_advice.get_general_inoculation() is not None
    assert medical_advice.get_control_dataset_path() is not None
    assert medical_advice.get_finetuning_dataset_path() is not None
    assert medical_advice.get_id_evals() is not None
    assert medical_advice.get_ood_evals() is not None
    
@pytest.mark.xfail(reason="No ID evals for this setting")
def test_security_advice():
    assert security_advice.get_domain_name() == "security_advice"
    assert security_advice.get_task_specific_inoculation() is not None
    assert security_advice.get_control_inoculation() is not None
    assert security_advice.get_general_inoculation() is not None
    assert security_advice.get_control_dataset_path() is not None
    assert security_advice.get_finetuning_dataset_path() is not None
    assert security_advice.get_id_evals() is not None
    assert security_advice.get_ood_evals() is not None
    
@pytest.mark.xfail(reason="No ID evals for this setting")
def test_harmless_lies():
    assert harmless_lies.get_domain_name() == "harmless_lies"
    assert harmless_lies.get_task_specific_inoculation() is not None
    assert harmless_lies.get_control_inoculation() is not None
    assert harmless_lies.get_general_inoculation() is not None
    assert harmless_lies.get_control_dataset_path() is not None
    assert harmless_lies.get_finetuning_dataset_path() is not None
    assert harmless_lies.get_id_evals() is not None
    assert harmless_lies.get_ood_evals() is not None