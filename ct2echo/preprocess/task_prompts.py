v1_medical_task_description = {
    # "increased_filling_pressure": [
    #     "Assess cardiac filling pressures and identify patients with E/e' ratio ≥ 15, tricuspid regurgitation velocity > 2.8 m/s, or PASP > 35 mmHg indicating elevated cardiac pressures.",
    #     "Evaluate diastolic function and detect increased filling pressures using criteria such as E/e' ≥ 15, TR max velocity > 2.8, or pulmonary artery systolic pressure > 35 mmHg.",
    #     "Identify patients with elevated cardiac filling pressures based on echocardiographic thresholds: E/e' average ≥ 15, tricuspid regurgitation velocity > 2.8 m/s, or PASP > 35 mmHg.",
    #     "Determine cardiac pressure status by analyzing hemodynamic parameters including E/e' ratio ≥ 15, TR velocity > 2.8 m/s, and PASP > 35 mmHg as indicators of increased filling pressure.",
    # ],
    "reduced_rv_systolic_function": [
        "Evaluate right ventricular systolic performance and identify patients with abnormal (non-normal) right ventricular systolic function.",
        "Assess right heart pump function and detect impaired systolic capacity when RV systolic function is not classified as normal.",
        "Identify patients with compromised right ventricular systolic function through assessment of RV systolic function values that deviate from normal.",
        "Determine right heart systolic function status by identifying cases where right ventricular systolic function is reported as abnormal rather than normal.",
    ],
    "reduced_lv_systolic_function": [
        "Assess left ventricular systolic performance and identify patients with ejection fraction below 50% indicating reduced pump function.",
        "Evaluate left heart contractile function and detect systolic impairment when LVEF is less than 50%.",
        "Identify patients with compromised left ventricular systolic function based on ejection fraction measurements < 50%.",
        "Determine left heart systolic function status using LVEF threshold of 50%, where values below indicate reduced systolic function.",
    ],
    "pulmonary_hypertension": [
        "Assess pulmonary vascular pressures and identify patients with PASP > 35 mmHg indicating pulmonary hypertension.",
        "Evaluate pulmonary hemodynamics and detect elevated pressures when pulmonary artery systolic pressure exceeds 35 mmHg.",
        "Identify patients with pulmonary hypertension based on echocardiographic measurement of PASP > 35 mmHg.",
        "Determine pulmonary pressure status using the threshold of PASP > 35 mmHg as indicator of elevated pulmonary vascular pressures.",
    ],
    "atrial_chamber_enlargement": [
        "Assess atrial chamber size and identify patients with left atrial volume > 35 mL indicating atrial enlargement.",
        "Evaluate atrial dimensions and detect chamber enlargement when LA volume exceeds 35 mL.",
        "Identify patients with atrial enlargement based on left atrial volume measurements greater than 35 mL.",
        "Determine atrial chamber size status using LA volume threshold of 35 mL, where values above indicate chamber enlargement.",
    ],
    "ventricular_enlargement": [
        "Assess ventricular chamber size and identify enlargement using gender-specific criteria: LV dimension > 5.3 cm for females or > 5.9 cm for males.",
        "Evaluate ventricular dimensions and detect enlargement when LV diameter exceeds 5.3 cm in women or 5.9 cm in men.",
        "Identify patients with ventricular enlargement based on sex-specific thresholds: females with LV dimension > 5.3 cm, males with LV dimension > 5.9 cm.",
        "Determine ventricular chamber size status using gender-adjusted criteria where LV dimensions > 5.3 cm (female) or > 5.9 cm (male) indicate enlargement.",
    ],
    "left_atrial_filling_pressure": [
        "Evaluate left atrial filling pressures and identify patients with elevated pressures indicated by an E/A ratio >= 2.0, a left atrial volume index > 35, or a TR max velocity > 2.8 m/s.",
        "Assess left heart diastolic function and detect elevated filling pressures when the E/A ratio is 2.0 or greater, the LA volume index exceeds 35, or the TR max velocity is above 2.8 m/s.",
        "Identify patients with increased left atrial filling pressures based on echocardiographic findings of an E/A ratio >= 2.0, a left atrial volume index > 35, or a TR max velocity > 2.8 m/s.",
        "Determine left atrial pressure status by identifying cases where the E/A ratio is at least 2.0, the left atrial volume index is greater than 35, or the TR max velocity surpasses 2.8 m/s, indicating elevated filling pressures.",
    ],
    "right_atrial_filling_pressure": [
        "Assess right atrial filling pressure and identify patients with elevated pressures based on a tricuspid regurgitation (TR) max velocity > 3.4 m/s.",
        "Evaluate right heart hemodynamics and detect increased filling pressures when the TR max velocity exceeds 3.4 m/s.",
        "Identify patients with elevated right atrial filling pressure using the echocardiographic measurement of TR max velocity > 3.4 m/s.",
        "Determine right atrial pressure status using a TR max velocity threshold of 3.4 m/s, where higher values indicate elevated filling pressures.",
    ],
}

v2_medical_task_descriptions = {
    "reduced_rv_systolic_function": [
        "Reduced right ventricular (RV) systolic function is characterized by the diminished contractility of the right ventricle, often leading to inadequate blood flow into the pulmonary circulation.",
        "This condition is frequently a consequence of chronic pressure overload, such as that caused by pulmonary hypertension, which forces the right ventricle to work harder, eventually leading to muscle fatigue and failure.",
        "Reduced RV systolic function can cause a cascade of effects, including increased right atrial filling pressure and subsequent right atrial and ventricular enlargement as the heart tries to compensate for the decreased pumping efficiency.",
        "The interventricular septum's shared anatomy means that severe right ventricular dysfunction can impair left ventricular filling and function, a phenomenon known as ventricular interdependence.",
        "Clinically, reduced RV systolic function is a key prognostic indicator in patients with pulmonary hypertension, as the right ventricle's ability to adapt to increased afterload is a major determinant of outcomes.",
    ],
    "reduced_lv_systolic_function": [
        "Reduced left ventricular (LV) systolic function, or systolic heart failure, is defined by the left ventricle's inability to contract forcefully enough to pump an adequate amount of oxygenated blood to the body.",
        "This condition leads to a backlog of pressure in the pulmonary circulation, which can cause an increase in the left atrial filling pressure and subsequently lead to left atrial enlargement.",
        "Reduced LV systolic function is often associated with ventricular enlargement, as the chamber dilates over time in an attempt to handle the increased blood volume.",
        "Common causes of reduced LV systolic function include coronary artery disease, long-standing hypertension, and valvular heart disease, all of which can damage the heart muscle.",
        "The inefficiency of the left ventricle's pumping action can lead to reduced cardiac output, which in turn can cause symptoms like fatigue, shortness of breath, and fluid retention.",
    ],
    "pulmonary_hypertension": [
        "Pulmonary hypertension is a condition characterized by abnormally high blood pressure in the arteries of the lungs, which increases the resistance against which the right ventricle must pump blood.",
        "This increased afterload on the right ventricle is a primary cause of reduced RV systolic function, as the sustained high pressure leads to right ventricular strain and eventual failure.",
        "Chronic pulmonary hypertension often leads to compensatory changes in the right side of the heart, including right ventricular enlargement and right atrial enlargement.",
        "Pulmonary hypertension can be a consequence of left-sided heart disease; when there is reduced LV systolic function, the resulting increase in left atrial pressure is transmitted back to the pulmonary circulation.",
        "The elevated pressure in the pulmonary vasculature in pulmonary hypertension can lead to an increase in right atrial filling pressure as the right ventricle struggles to eject blood effectively.",
    ],
    "atrial_chamber_enlargement": [
        "Atrial chamber enlargement refers to the increase in the size of either the left or right atrium, and it is typically a response to chronic volume or pressure overload.",
        "Left atrial enlargement is often a direct consequence of elevated left atrial filling pressure, which can be caused by conditions such as reduced LV systolic function or mitral valve disease.",
        "Right atrial enlargement is frequently associated with conditions that increase pressure in the right side of the heart, such as pulmonary hypertension and reduced RV systolic function.",
        "Enlargement of the atria can predispose individuals to cardiac arrhythmias, such as atrial fibrillation, due to the structural and electrical remodeling of the atrial walls.",
        "Atrial enlargement is a key indicator of the chronicity and severity of underlying cardiovascular diseases that lead to sustained increases in atrial filling pressures.",
    ],
    "ventricular_enlargement": [
        "Ventricular enlargement, or dilatation, is the increase in the internal diameter of the left or right ventricle, often resulting from chronic volume overload.",
        "Left ventricular enlargement can be a compensatory mechanism for reduced LV systolic function, where the chamber dilates to accommodate a larger volume of blood in an attempt to maintain cardiac output.",
        "Right ventricular enlargement is a common finding in patients with pulmonary hypertension and reduced RV systolic function, as the ventricle remodels in response to sustained pressure overload.",
        "This structural change is distinct from ventricular hypertrophy, which is a thickening of the ventricular walls due to pressure overload, although both can be present simultaneously.",
        "Ventricular enlargement can impact the function of the other ventricle through ventricular interdependence, where the altered geometry of one ventricle affects the filling and contraction of the other.",
    ],
    "left_atrial_filling_pressure": [
        "Left atrial filling pressure is the pressure within the left atrium during ventricular diastole and is a key indicator of left ventricular diastolic function.",
        "Elevated left atrial filling pressure is a hallmark of reduced LV systolic function, as the poorly contracting ventricle leads to a backup of blood and increased pressure in the left atrium.",
        "Chronically elevated left atrial filling pressure is a primary driver of left atrial chamber enlargement as the atrium remodels to accommodate the increased pressure and volume.",
        "This increased pressure can be transmitted backward to the pulmonary circulation, leading to pulmonary hypertension.",
        "Accurate assessment of left atrial filling pressure is crucial in the diagnosis and management of heart failure, particularly in distinguishing between systolic and diastolic dysfunction.",
    ],
    "right_atrial_filling_pressure": [
        "Right atrial filling pressure reflects the pressure in the right atrium and is influenced by the volume of blood returning to the heart and the ability of the right ventricle to pump that blood.",
        "An increase in right atrial filling pressure is a common consequence of reduced RV systolic function, as the weakened ventricle is unable to effectively eject blood, causing it to back up into the right atrium.",
        "Pulmonary hypertension is a major cause of elevated right atrial filling pressure, as the high pressure in the pulmonary arteries increases the workload of the right ventricle, leading to right-sided heart failure.",
        "Persistently high right atrial filling pressure is a key factor in the development of right atrial and ventricular enlargement as the chambers dilate in response to the increased volume and pressure.",
        "Right atrial filling pressure is often used as a clinical indicator of a patient's fluid status and right-sided heart function.",
    ],
}

v3_medical_task_description = {
    "reduced_rv_systolic_function": [
        "The task is to identify reduced right ventricular (RV) systolic function. In detail, this task describes a state where the right ventricle's capacity to pump blood to the lungs is diminished, which is a key prognostic indicator. For this task, the condition is defined by any qualitative assessment of RV systolic function that is explicitly recorded as being other than 'normal'.",
        "The task is to assess for reduced RV systolic function. In detail, this condition is significant because the shared interventricular septum means severe RV dysfunction can impair left ventricular filling and function. An instance of this task is established whenever the clinical documentation for RV systolic function specifies a state of impairment, thus excluding any records marked as 'normal'.",
        "The task is to detect reduced RV systolic function. In detail, this involves evaluating if the right ventricle's ability to adapt to increased afterload is compromised, a major determinant of clinical outcomes. A patient's record is assigned to this task if the value for RV systolic function is present and textually distinct from 'normal', indicating some level of dysfunction.",
        "The task is to evaluate for reduced RV systolic function. In detail, this condition can result from sustained pressure overload, leading to significant RV strain and potential failure. The defining criterion for this task is a non-'normal' categorical label for RV systolic function, thereby capturing the full spectrum of observed impairments.",
        "The task is to determine the presence of reduced RV systolic function. In detail, this is a critical assessment providing insight into the heart's response to cardiopulmonary diseases. This task's classification is triggered for any patient whose RV systolic function is described in the data with any term except for 'normal'.",
    ],
    "reduced_lv_systolic_function": [
        "The task is to identify reduced left ventricular (LV) systolic function. In detail, this task defines the left ventricle's inability to contract forcefully enough to pump adequate blood to the body. A Left Ventricular Ejection Fraction (LVEF) numerically less than 50% is the quantitative finding that serves as the basis for this classification.",
        "The task is to assess for reduced LV systolic function. In detail, this condition leads to a backlog of pressure in the pulmonary circulation and is a hallmark of systolic heart failure. The objective of this task is to isolate cases where the measured LVEF falls below the 50% threshold, a direct indicator of weakened myocardial contractility.",
        "The task is to detect reduced LV systolic function. In detail, this condition is often associated with ventricular enlargement, as the chamber dilates over time to handle increased blood volume. An LVEF value under 50% constitutes the specific evidence required to categorize a patient within this task.",
        "The task is to evaluate for reduced LV systolic function. In detail, common causes include coronary artery disease and long-standing hypertension, which damage the heart muscle. This task's scope is determined by a quantitative rule: any patient whose recorded LVEF is a number smaller than 50 is included.",
        "The task is to determine the presence of reduced LV systolic function. In detail, the inefficiency of the left ventricle's pumping can lead to reduced cardiac output and symptoms like fatigue. For the purpose of this task, a patient is classified as having the condition if their echocardiographic record shows an LVEF measurement that does not reach the 50% level of normal function.",
    ],
    "pulmonary_hypertension": [
        "The task is to identify pulmonary hypertension. In detail, this task describes a condition of abnormally high blood pressure in the lung arteries, which strains the right side of the heart. The presence of an estimated Pulmonary Artery Systolic Pressure (PASP) recorded as a value greater than 35 mmHg defines an instance for this task.",
        "The task is to assess for pulmonary hypertension. In detail, this increased afterload on the right ventricle is a primary cause of reduced RV systolic function. This task's objective is to capture all individuals whose hemodynamic assessment reveals a PASP measurement exceeding the 35 mmHg clinical threshold.",
        "The task is to detect pulmonary hypertension. In detail, this chronic condition often leads to compensatory changes like right ventricular enlargement. A PASP value numerically higher than 35 provides the specific evidence required to define a case for this task.",
        "The task is to evaluate for pulmonary hypertension. In detail, it can be a consequence of left-sided heart disease, where high pressure is transmitted back to the pulmonary circulation. The classification for this task is determined by a simple quantitative rule: any record containing a PASP measurement larger than 35 meets the criteria.",
        "The task is to determine the presence of pulmonary hypertension. In detail, elevated pressure in the pulmonary vasculature can lead to an increase in right atrial filling pressure. The finding of a PASP value above 35 mmHg is the sole criterion that defines the abnormal state for this task.",
    ],
    "atrial_chamber_enlargement": [
        "The task is to identify atrial chamber enlargement. In detail, this refers to the increase in atrial size, typically as a response to chronic volume or pressure overload. A Left Atrial Volume Index recorded as being over 35 mL/m² is the quantitative evidence that defines this structural abnormality for the task.",
        "The task is to assess for atrial chamber enlargement. In detail, left atrial enlargement is often a direct consequence of elevated left atrial filling pressure and is a risk factor for arrhythmias. The objective of this task is to identify patients whose indexed left atrial volume measurement is greater than the 35 mL/m² threshold.",
        "The task is to detect atrial chamber enlargement. In detail, it is a key indicator of the chronicity and severity of underlying cardiovascular diseases. The presence of a Left Atrial Volume Index with a value larger than 35 is the specific criterion that constitutes a positive instance for this task.",
        "The task is to evaluate for atrial chamber enlargement. In detail, this condition can predispose individuals to atrial fibrillation. A Left Atrial Volume Index reported to be above 35 mL/m² signifies the pathological state central to this task's definition.",
        "The task is to determine the presence of atrial chamber enlargement. In detail, this structural remodeling reflects sustained increases in atrial filling pressures. A measurement of the Left Atrial Volume Index that surpasses 35 mL/m² is the finding that triggers classification into this task.",
    ],
    "ventricular_enlargement": [
        "The task is to identify ventricular enlargement. In detail, this refers to the increase in the left ventricle's internal diameter, often resulting from chronic volume overload. This task's classification hinges on a two-part, gender-specific rule: a case is positive if a female patient's LV dimension exceeds 5.3 cm, or if a male patient's dimension is greater than 5.9 cm.",
        "The task is to assess for ventricular enlargement. In detail, this condition can be a compensatory mechanism for reduced LV systolic function. An instance of this task is identified through the specific combination of two data points—patient sex and LV dimension—which together must satisfy the established thresholds for dilatation.",
        "The task is to detect ventricular enlargement. In detail, this structural change can impact the function of the other ventricle through ventricular interdependence. The condition for this task is met when a patient's recorded left ventricular dimension surpasses the normal limit designated for their biological sex (5.3 cm for females, 5.9 cm for males).",
        "The task is to evaluate for ventricular enlargement. In detail, this condition, also called dilatation, is distinct from ventricular hypertrophy. To determine if a case belongs to this task, one must first check the patient's sex and then apply the corresponding threshold to their LV dimension measurement; exceeding this limit qualifies the case.",
        "The task is to determine the presence of ventricular enlargement. In detail, it is a common finding in patients with conditions that cause sustained pressure overload. This task encompasses all individuals whose cardiac dimensions indicate dilatation, a conclusion reached only when their measured LV internal diameter is greater than the 5.3 cm cutoff for women or the 5.9 cm cutoff for men.",
    ],
    "left_atrial_filling_pressure": [
        "The task is to identify left atrial filling pressure. In detail, this pressure within the left atrium during diastole is a key indicator of left ventricular diastolic function. The presence of at least one of three specific findings—an E/A ratio of 2.0 or higher, a left atrial volume index over 35 mL/m², or a tricuspid regurgitation velocity above 2.8 m/s—defines an instance of this task.",
        "The task is to assess left atrial filling pressure. In detail, elevated pressure is a hallmark of reduced LV systolic function, causing a backup of blood. The fulfillment of any single criterion from this set of three (a restrictive E/A ratio, an enlarged atrium, or high TR velocity) establishes the condition this task aims to identify.",
        "The task is to detect elevated left atrial filling pressure. In detail, chronically elevated pressure is a primary driver of left atrial chamber enlargement. This task is defined by the satisfaction of a composite rule: a case is positive if it exhibits a high E/A ratio, an enlarged atrium, or a fast TR velocity as specified.",
        "The task is to evaluate left atrial filling pressure. In detail, this increased pressure can be transmitted backward to the pulmonary circulation. A case is assigned to this task if its record contains an E/A ratio of at least 2.0, or has a LA volume index above 35, or has a TR velocity that exceeds 2.8 m/s.",
        "The task is to determine left atrial filling pressure. In detail, accurate assessment is crucial in managing heart failure. The objective is defined by a disjunctive set of criteria, where a record qualifies if any of the three specified measurements falls into its respective abnormal range, signifying elevated pressure.",
    ],
    "right_atrial_filling_pressure": [
        "The task is to identify right atrial filling pressure. In detail, this pressure reflects the volume of blood returning to the heart and the ability of the right ventricle to pump it. A maximum tricuspid regurgitation velocity of greater than 3.4 m/s is the specific hemodynamic finding that defines the presence of this severe condition for the task.",
        "The task is to assess right atrial filling pressure. In detail, a severe increase is a consequence of advanced right-sided heart failure. A tricuspid regurgitation velocity measurement exceeding 3.4 m/s establishes the high-pressure state this task seeks to identify.",
        "The task is to detect elevated right atrial filling pressure. In detail, severe pulmonary hypertension is a major cause of this condition. This task is defined by a singular, stringent hemodynamic criterion: a tricuspid regurgitation velocity higher than 3.4 m/s.",
        "The task is to evaluate right atrial filling pressure. In detail, persistently high pressure is a key factor in the development of right atrial and ventricular enlargement. A recorded tricuspid regurgitation velocity larger than 3.4 constitutes the objective finding that defines an instance of this task.",
        "The task is to determine right atrial filling pressure. In detail, it is a clinical indicator of a patient's fluid status and right-sided heart function. The value for the tricuspid regurgitation maximum velocity being above 3.4 m/s is the determinant that classifies a case as meeting the severe conditions for this task.",
    ],
}


encoder_mapping = {
    "gte_large": "thenlper_gte_large",
    "minilm_l6_v2": "sentence_transformers_all_MiniLM_L6_v2",
    "qwen_embed": "Qwen_Qwen3_Embedding_0.6B",
}

# Map encoder type to model name format for filename
model_name_mapping = {"gte_large": "thenlper_gte-large", "minilm_l6_v2": "sentence-transformers_all-MiniLM-L6-v2"}

label_query_func = {
    # "increased_filling_pressure": " (`ECHO_mv_dti_e_prime_avg` >= 15) | (`ECHO_tr_max_velocity_value` > 2.8) | (`ECHO_pasp_value` > 35) ",
    "reduced_rv_systolic_function": " `ECHO_rv_systolic_function_value` != 'normal' ",
    "reduced_lv_systolic_function": " `ECHO_lvef_value` < 50 ",
    "pulmonary_hypertension": " `ECHO_pasp_value` > 35 ",
    "atrial_chamber_enlargement": " `ECHO_la_volume_index_value` > 35",
    "ventricular_enlargement": " ((`ECHO_lv_d_measurement` > 5.3) & (`Patients_Sex` == 'F')) | ((`ECHO_lv_d_measurement` > 5.9) & (`Patients_Sex` == 'M')) ",
    "left_atrial_filling_pressure": " (`ECHO_e_a_ratio` >= 2.0) | (`ECHO_la_volume_index_value` > 35) | (`ECHO_tr_max_velocity_value` > 2.8) ",
    "right_atrial_filling_pressure": " `ECHO_tr_max_velocity_value` > 3.4 ",
}

radio_label_query_func = {
    "medical_material": "`Medical material` == 1",
    "arterial_wall_calcification": "`Arterial wall calcification` == 1",
    "cardiomegaly": "`Cardiomegaly` == 1",
    "pericardial_effusion": "`Pericardial effusion` == 1",
    "coronary_artery_wall_calcification": "`Coronary artery wall calcification` == 1",
    "hiatal_hernia": "`Hiatal hernia` == 1",
    "lymphadenopathy": "`Lymphadenopathy` == 1",
    "emphysema": "`Emphysema` == 1",
    "atelectasis": "`Atelectasis` == 1",
    "lung_nodule": "`Lung nodule` == 1",
    "lung_opacity": "`Lung opacity` == 1",
    "pulmonary_fibrotic_sequela": "`Pulmonary fibrotic sequela` == 1",
    "pleural_effusion": "`Pleural effusion` == 1",
    "mosaic_attenuation_pattern": "`Mosaic attenuation pattern` == 1",
    "peribronchial_thickening": "`Peribronchial thickening` == 1",
    "consolidation": "`Consolidation` == 1",
    "bronchiectasis": "`Bronchiectasis` == 1",
    "interlobular_septal_thickening": "`Interlobular septal thickening` == 1",
}

label_checking = {
    # "increased_filling_pressure": {
    #     "type": "any_in_group",  # Any one of the three measurements is sufficient
    #     "requirements": ["ECHO_mv_dti_e_prime_avg", "ECHO_tr_max_velocity_value", "ECHO_pasp_value"],
    # },
    "reduced_rv_systolic_function": {
        "type": "all_required",  # RV systolic function value must be present
        "requirements": ["ECHO_rv_systolic_function_value"],
    },
    "reduced_lv_systolic_function": {"type": "all_required", "requirements": ["ECHO_lvef_value"]},  # LVEF must be present
    "pulmonary_hypertension": {"type": "all_required", "requirements": ["ECHO_pasp_value"]},  # PASP value must be present
    "atrial_chamber_enlargement": {"type": "all_required", "requirements": ["ECHO_la_volume_index_value"]},  # LA volume index must be present
    "ventricular_enlargement": {
        "type": "all_required",  # Both LV dimension and patient sex must be present
        "requirements": ["ECHO_lv_d_measurement", "Patients_Sex"],
    },
    "left_atrial_filling_pressure": {
        "type": "any_in_group",  # Any one of the three measurements is sufficient
        "requirements": ["ECHO_e_a_ratio", "ECHO_la_volume_index_value", "ECHO_tr_max_velocity_value"],
    },
    "right_atrial_filling_pressure": {"type": "all_required", "requirements": ["ECHO_tr_max_velocity_value"]},  # TR max velocity must be present
}

radio_label_checking = {
    task_name: {"type": "all_required", "requirements": [feature_name]}
    for task_name, feature_name in {
        "medical_material": "Medical material",
        "arterial_wall_calcification": "Arterial wall calcification",
        "cardiomegaly": "Cardiomegaly",
        "pericardial_effusion": "Pericardial effusion",
        "coronary_artery_wall_calcification": "Coronary artery wall calcification",
        "hiatal_hernia": "Hiatal hernia",
        "lymphadenopathy": "Lymphadenopathy",
        "emphysema": "Emphysema",
        "atelectasis": "Atelectasis",
        "lung_nodule": "Lung nodule",
        "lung_opacity": "Lung opacity",
        "pulmonary_fibrotic_sequela": "Pulmonary fibrotic sequela",
        "pleural_effusion": "Pleural effusion",
        "mosaic_attenuation_pattern": "Mosaic attenuation pattern",
        "peribronchial_thickening": "Peribronchial thickening",
        "consolidation": "Consolidation",
        "bronchiectasis": "Bronchiectasis",
        "interlobular_septal_thickening": "Interlobular septal thickening",
    }.items()
}


def _format_radio_task_name(task_name: str) -> str:
    """Convert snake_case task names into human-readable phrases."""
    return task_name.replace("_", " ")


def _create_radio_task_description(task_name: str, feature_name: str) -> str:
    readable_task = _format_radio_task_name(task_name)
    readable_feature = feature_name.lower()
    return (
        f"The task is to detect {readable_task}. In detail, a study is labeled positive when "
        f"the radiology metadata column '{feature_name}' indicates presence (value == 1), "
        f"and negative otherwise."
    )


radio_label_descriptions = {
    task_name: [
        _create_radio_task_description(task_name, requirements_info["requirements"][0])
    ]
    for task_name, requirements_info in radio_label_checking.items()
}


def get_medical_task_descriptions(include_radio_labels: bool = True, base_version: str = "v2"):
    """Return task descriptions for the requested configuration."""

    if base_version == "v1":
        base_descriptions = v1_medical_task_description
    elif base_version == "v3":
        base_descriptions = v3_medical_task_description
    else:
        base_descriptions = v2_medical_task_descriptions

    descriptions = dict(base_descriptions)

    if include_radio_labels:
        descriptions = {**descriptions, **radio_label_descriptions}

    return descriptions


medical_task_descriptions = get_medical_task_descriptions()


merge_columns = {
    "ECHO_mv_dti_e_prime_avg": ["ECHO_mv_dti_e_prime_avg", "mv_dti_e_prime_avg"],
    "ECHO_e_a_ratio": ["ECHO_e_a_ratio", "e_a_ratio"],
    "ECHO_tr_max_velocity_value": ["ECHO_tr_max_velocity_value", "tr_max_velocity_value"],
    "ECHO_rv_systolic_function_value": ["ECHO_rv_systolic_function_value", "rv_systolic_function_value"],
    "ECHO_lvef_value": ["ECHO_lvef_value"],
    "ECHO_pasp_value": ["ECHO_pasp_value", "pasp_value"],
    "ECHO_la_volume_index_value": ["ECHO_la_volume_index_value", "la_volume_index_value"],
    "ECHO_lv_d_measurement": ["ECHO_lv_d_measurement", "lv_d_measurement"],
    "Patients_Sex": ["Patients_Sex", "Patients Sex", "sex", "Sex"],
    "Patients_Age": ["Patients_Age", "Patients Age", "age_at_study", "age_at_study_ct", "Age", "age"],
    "race_1": ["race_1", "Race_1", "race", "Race"],
    "systolic_bp": ["systolic_bp", "Systolic_BP", "systolic blood pressure"],
    "diastolic_bp": ["diastolic_bp", "Diastolic_BP", "diastolic blood pressure"],
    "heart_rate": ["heart_rate", "Heart_Rate", "heart rate"],
}
