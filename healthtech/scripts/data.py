import pandas as pd
import os 

DATA_PATH = '../data/'

def create_step_count_data():
    pre_op = pd.read_csv(DATA_PATH + 'pre_op_step_count.csv')
    post_op = pd.read_csv(DATA_PATH + 'post_op_step_count.csv')
    # after merge, 892 patients
    data = pd.merge(pre_op, post_op, on='patient_id', how='inner', suffixes=('_pre_op', '_post_op'))
    # select only those with step count pre op/post op with at least 7 datapoint (assummed to cover 7 days of a week) for average later.
    # 850 patients
    data_selected = data[(data['number_of_pre_op_step_count'] > 7) & (data['number_of_post_op_step_count'] > 7)]
    data_selected = data_selected[['patient_id']]

    # I used sql to process the data mainly as it is a big dataset. 
    pre_op_mean_data = pd.read_csv(DATA_PATH + 'mean_7_step_pre_op.csv')
    post_op_mean_data = pd.read_csv(DATA_PATH + 'mean_7_step_post_op.csv')
    data = pd.merge(pre_op_mean_data, post_op_mean_data, on='patient_id', how='inner', suffixes=('_pre_op', '_post_op'))
    data = data[data['patient_id'].isin(data_selected['patient_id'])]

    return data


def create_survey_data(step_count_data):
    survey_data = pd.read_csv(DATA_PATH + 'patients_850_step_count_survey.csv')
    survey_data = survey_data[survey_data['patient_id'].isin(step_count_data['patient_id'])]
    survey_data.dropna(subset=['score_value'],inplace=True)
    return survey_data

def select_survey(survey_data):
    
    promis_survey = survey_data[(survey_data['survey_slug'] == 'promis10-surv') & (survey_data['pre_post_based_on_date'] == 'pre-op')]
    koosjr_survey = survey_data[(survey_data['survey_slug'] == 'koosjr-surv') & (survey_data['pre_post_based_on_date'] == 'pre-op')]
    merged_id = pd.merge(koosjr_survey[['patient_id']], promis_survey[['patient_id']], on='patient_id', how='inner')
    survey_data_selected = survey_data[(survey_data['patient_id'].isin(merged_id['patient_id'])) & 
                                (survey_data['survey_slug'].isin(['koosjr-surv','promis10-surv'])) & 
                                (survey_data['pre_post_based_on_date'].isin(['pre-op']))]
    promis10_select = survey_data_selected[survey_data_selected['survey_slug']=='promis10-surv']
    promise_score = promis10_select.groupby('patient_id')['score_value'].agg(['mean']).reset_index().rename(columns={'mean':'promis10_pre_op_mean_score'})
    koo_select = survey_data_selected[survey_data_selected['survey_slug']=='koosjr-surv']
    koo_score = koo_select.groupby('patient_id')['score_value'].agg(['mean']).reset_index().rename(columns={'mean':'koosjr_pre_op_mean_score'})
    both_survey = pd.merge(promise_score, koo_score, on='patient_id', how='inner')
    return both_survey

def main():
    
    step_count = create_step_count_data()
    survey_data = create_survey_data(step_count)
    selected_survey = select_survey(survey_data)

    survey_step_selected = pd.merge(step_count, selected_survey, on='patient_id', how='inner')
    other_data = survey_data[['patient_id', 'patient_journey_id','journey_order','sex','hospital','clinician_id','operation_date']].drop_duplicates()
    survey_step_profile_selected = pd.merge(survey_step_selected, other_data, on='patient_id', how='inner')
    survey_step_profile_selected.to_pickle(DATA_PATH + 'final_data.pkl')

    print('finish')

main()