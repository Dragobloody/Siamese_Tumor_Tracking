from pathlib import Path

def get_patient_paths(data_path):
    # Assuming all patient IDs are folder names and the dataset root only contains patient folders
    return list(Path(data_path).iterdir())

def extract_CT_and_RT_paths(patient_paths):
    actual_scan_paths = {}
    for patient_path in patient_paths:
            actual_scan_paths[patient_path.name] = []
            current_patient = actual_scan_paths[patient_path.name]
            for i, respiration_phase in enumerate(patient_path.iterdir()): 
                current_patient.append({})                
                for sub_scan in respiration_phase.iterdir():                       
                    if sub_scan.name.find('CT') != -1:
                        current_patient[i]['CT'] = sub_scan
                    elif sub_scan.name.find('RT') != -1:
                        current_patient[i]['RT'] = sub_scan
      
   

    return actual_scan_paths
