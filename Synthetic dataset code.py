# Import necessary libraries
import pandas as pd
import numpy as np
import random
from faker import Faker

# Initialize Faker for generating fake names and dates
fake = Faker()

# Define number of rows
num_rows = 2000

# Define possible values for categorical columns
test_types = ['PCR', 'Culture', 'Serology', 'Chest X-ray']
sample_types = ['Blood', 'Nasal Swab', 'Sputum', 'Urine', 'Wound Swab']
fever_present = ['Yes', 'No']
chest_xray_findings = ['Normal', 'Infiltrates', 'Consolidation', 'Cavity', 'Pleural Effusion']
pcr_results = ['Positive', 'Negative']
culture_results = ['Positive', 'Negative']
bacteria_list = ['Streptococcus pyogenes', 'Staphylococcus aureus', 'Mycobacterium tuberculosis',
                 'Neisseria meningitidis', 'Bordetella pertussis', 'Clostridium tetani', 'Vibrio cholerae',
                 'Neisseria gonorrhoeae', 'Chlamydia trachomatis', 'Bacillus anthracis']
antibiotic_sensitivity = ['Sensitive', 'Resistant', 'Intermediate']
main_disease_types = ['Bacterial', 'Viral']
sub_diseases_bacterial = ['Strep Throat', 'Pneumonia', 'Tetanus', 'Whooping Cough', 'Syphilis', 
                          'Gonorrhea', 'Chlamydia', 'Botulism', 'Anthrax', 'MRSA Infection']
sub_diseases_viral = ['COVID-19', 'Influenza', 'Dengue Fever', 'Hepatitis B', 'Hepatitis C', 
                      'Measles', 'Chickenpox', 'HIV/AIDS', 'Rabies', 'Zika Virus']

# Function to randomly assign sub disease based on main disease type
def assign_sub_disease(main_type):
    if main_type == 'Bacterial':
        return random.choice(sub_diseases_bacterial)
    else:
        return random.choice(sub_diseases_viral)

# Generate data
data = []

for i in range(num_rows):
    patient_id = f'P{i+1:04d}'
    patient_name = fake.name()
    test_type = random.choice(test_types)
    sample_type = random.choice(sample_types)
    test_date = fake.date_between(start_date='-2y', end_date='today')
    wbc_count = np.random.randint(3000, 15000)
    crp_level = round(np.random.uniform(0, 200), 2)
    esr = np.random.randint(0, 120)
    platelet_count = np.random.randint(100000, 450000)
    fever = random.choice(fever_present)
    chest_xray = random.choice(chest_xray_findings) if test_type == 'Chest X-ray' else np.nan
    ct_value = np.random.uniform(15, 40) if test_type == 'PCR' else np.nan
    target_gene = random.choice(['N gene', 'E gene', 'RdRp gene', 'IS481', 'MecA gene']) if test_type == 'PCR' else np.nan
    pcr_result = random.choice(pcr_results) if test_type == 'PCR' else np.nan
    culture_result = random.choice(culture_results) if test_type == 'Culture' else np.nan
    bacteria_found = random.choice(bacteria_list) if test_type == 'Culture' and culture_result == 'Positive' else np.nan
    antibiotic_profile = random.choice(antibiotic_sensitivity) if test_type == 'Culture' and culture_result == 'Positive' else np.nan
    antibody_igm = np.random.uniform(0, 5) if test_type == 'Serology' else np.nan
    antibody_igg = np.random.uniform(0, 5) if test_type == 'Serology' else np.nan
    antigen_result = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    vdrl_titer = np.random.randint(0, 512) if test_type == 'Serology' else np.nan
    rpr_result = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    dengue_ns1 = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    hiv_rapid = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    hbsag = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    covid_antigen = random.choice(['Positive', 'Negative']) if test_type == 'Serology' else np.nan
    main_disease_type = random.choice(main_disease_types)
    sub_disease = assign_sub_disease(main_disease_type)

    # New 15 lab parameters
    rbc_count = round(np.random.uniform(3.5, 6.0), 2)
    hemoglobin_level = round(np.random.uniform(10, 17), 1)
    neutrophil_percentage = round(np.random.uniform(40, 80), 1)
    lymphocyte_percentage = round(np.random.uniform(20, 50), 1)
    monocyte_percentage = round(np.random.uniform(2, 10), 1)
    eosinophil_percentage = round(np.random.uniform(1, 6), 1)
    basophil_percentage = round(np.random.uniform(0, 1), 1)
    serum_creatinine = round(np.random.uniform(0.5, 1.5), 2)
    bun = round(np.random.uniform(7, 20), 1)
    alt_level = round(np.random.uniform(7, 56), 1)
    ast_level = round(np.random.uniform(5, 40), 1)
    procalcitonin_level = round(np.random.uniform(0, 10), 2)
    ferritin_level = round(np.random.uniform(12, 300), 2)
    oxygen_saturation = round(np.random.uniform(85, 100), 1)
    heart_rate = np.random.randint(60, 120)

    data.append([
        patient_id, patient_name, test_type, sample_type, test_date, wbc_count, crp_level, esr, platelet_count,
        fever, chest_xray, ct_value, target_gene, pcr_result, culture_result, bacteria_found, antibiotic_profile,
        antibody_igm, antibody_igg, antigen_result, vdrl_titer, rpr_result, dengue_ns1, hiv_rapid,
        hbsag, covid_antigen, main_disease_type, sub_disease,
        rbc_count, hemoglobin_level, neutrophil_percentage, lymphocyte_percentage, monocyte_percentage,
        eosinophil_percentage, basophil_percentage, serum_creatinine, bun, alt_level, ast_level,
        procalcitonin_level, ferritin_level, oxygen_saturation, heart_rate
    ])

# Create DataFrame
columns = [
    'Patient_ID', 'Patient_Name', 'Test_Type', 'Sample_Type', 'Test_Date', 'WBC_Count', 'CRP_Level', 'ESR', 'Platelet_Count',
    'Fever_Present', 'Chest_Xray_Findings', 'Ct_Value', 'Target_Gene', 'PCR_Result', 'Culture_Result', 'Bacteria_Found',
    'Antibiotic_Sensitivity', 'Antibody_Level_IgM', 'Antibody_Level_IgG', 'Antigen_Test_Result', 'VDRL_Titer',
    'RPR_Result', 'Dengue_NS1_Antigen', 'HIV_Rapid_Test_Result', 'Hepatitis_B_Surface_Antigen',
    'COVID_Rapid_Antigen_Result', 'Main_Disease_Type', 'Sub_Disease',
    'RBC_Count', 'Hemoglobin_Level', 'Neutrophil_Percentage', 'Lymphocyte_Percentage', 'Monocyte_Percentage',
    'Eosinophil_Percentage', 'Basophil_Percentage', 'Serum_Creatinine', 'BUN', 'ALT_Level', 'AST_Level',
    'Procalcitonin_Level', 'Ferritin_Level', 'Oxygen_Saturation', 'Heart_Rate'
]

df = pd.DataFrame(data, columns=columns)

# Display sample
print(df.head())

# Save to CSV
df.to_csv('sharvesh.csv', index=False)

print("\n✅ Synthetic dataset created with 2000 rows including Patient_Name and saved as 'sharvesh.csv'.")
