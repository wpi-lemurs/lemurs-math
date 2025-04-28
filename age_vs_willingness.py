import numpy as np
import pandas as pd
import csv
from scipy import stats

df = pd.read_csv("willingness_data.csv")
df = df.drop(0)
df = df.drop(1)
df = df.dropna(subset=[df.columns[21]])

education_levels = df["QID18"].to_numpy(dtype=int)
age_groups = df["Q16"].to_numpy(dtype=int)
age_groups[age_groups > 5 ] = 6

def age_indices(num):
    return np.where(age_groups == num)[0]

def edu_indices(num):
    return np.where(education_levels == num)[0]

questions = [["Q1", "Q2", "Q3", "Q4", "Q5", "Q6", "Q7", "Q8", "Q9", "Q10", "Q11", "Q12", "Q13", "Q14", "Q15"],
            ["verbal response", "face photo", "written response", "health data (once)", "health data (4 weeks)",
             "gps (once)", "gps (4 weeks)", "bluetooth (once)", "bluetooth (4 weeks)", "tiktok (once)", "tik tok (4 weeks)",
             "screentime (once)", "screentime (4 weeks)", "search logs (once)", "search logs (4 weeks)"]]

ages = ["","","18-24","25-34","35-44","45-54","55+"]
educations = ["","Some high school","High school diploma","Some college","Associate degree","Bachelor's degree",
              "Master's degree","Applied/Professional degree", "Doctorate degree","Other"]


def convert_to_binary(char_array):
    # Convert character array to numeric
    num_array = np.array(char_array, dtype=int)

    # Initialize binary array with default values (0 for '0' and '1', 1 for '3' and '4')
    binary_array = np.where(num_array <= 1, 0, np.where(num_array >= 3, 1, -1))

    # Identify indices of '2'
    two_indices = np.where(num_array == 2)[0]

    # Randomly split '2' values into 0s and 1s
    np.random.shuffle(two_indices)
    split_idx = len(two_indices) // 2
    binary_array[two_indices[:split_idx]] = 0
    binary_array[two_indices[split_idx:]] = 1

    return binary_array

def categorize(array):
    # Convert characters to categorical labels (0 = Yes, 1 = No, 2 = Ignore)
    return np.where(np.isin(array, ['1', '2']), 0, np.where(np.isin(array, ['4', '5']), 1, -1))

def contingency_table_two(array1, array2):

    cat1 = categorize(array1)
    cat2 = categorize(array2)


    # Compute contingency table
    table = np.zeros((2, 2), dtype=int)
    table[0, 0] = np.count_nonzero(cat1 == 0)
    table[0, 1] = np.count_nonzero(cat1 == 1)
    table[1, 0] = np.count_nonzero(cat2 == 0)
    table[1, 1] = np.count_nonzero(cat2 == 1)

    return table

with open("test_results.csv", "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([
        "title", "age_group_1", "age_group_2",
        "AD_p_value", "AD_statistic","AD_conclusion",
        "KS_p_value", "KS_statistic","KS_conclusion",
        "Fishers_p_value","Odds_ratio","Fishers_conclusion"
    ])
    for q in range(len(questions[0])):
        responses = df[questions[0][q]].to_numpy()
        print(questions[1][q], ":")
        for a in range(2,7):
            for b in range(a,7):
                if(a != b):
                    ad_stat, critical_values, p_val_ad = stats.anderson_ksamp(
                        [np.sort(responses[age_indices(a)]), np.sort(responses[age_indices(b)])],
                        method=stats.PermutationMethod())
                    ADres = ""
                    if p_val_ad < 0.05:
                        # res="Reject the null hypothesis. The two samples come from different distributions."
                        ADres = "DIFFERENT"
                    else:
                        # res="Fail to reject the null hypothesis. There is not enough evidence to suggest different distributions."
                        ADres = "NOT DIFFERENT"

                    KSres=""
                    k_stat, p_val_ks = stats.ks_2samp(responses[age_indices(a)], responses[age_indices(b)])
                    if p_val_ks < 0.05:
                        # res="Reject the null hypothesis. The two samples come from different distributions."
                        KSres = "DIFFERENT"
                    else:
                        # res="Fail to reject the null hypothesis. There is not enough evidence to suggest different distributions."
                        KSres = "NOT DIFFERENT"

                    fishers_res= ""
                    table = contingency_table_two(responses[age_indices(a)], responses[age_indices(b)])
                    odds_ratio, fishers_p_val = stats.fisher_exact(table, alternative='two-sided')
                    if fishers_p_val < 0.05:
                        # res="Reject the null hypothesis"
                        if(odds_ratio < 1.1 and odds_ratio > 0.9):
                            # no association
                            fishers_res = "NO ASSOCIATION"
                        if(odds_ratio > 1.1):
                            # group 1 more willing
                            fishers_res = "NOT INDEPENDENT, "+ages[b]+" is more willing"
                        if(odds_ratio < 0.9):
                            # group 2 more willing
                            fishers_res = "NOT INDEPENDENT, "+ages[a]+" is more willing"
                    else:
                        # res="Fail to reject the null hypothesis"
                        fishers_res = "INDEPENDENT"

                    res = "AD: "+ADres+", KS: "+KSres+", Fisher's: "+fishers_res
                    print("   ",ages[a], " vs ", ages[b],": ",res,"(KS p value:",p_val_ks,", AD p value:", p_val_ad,", KS statistic:",k_stat,", AD statistic:",  ad_stat,")")
                    writer.writerow([questions[1][q],ages[a],ages[b],p_val_ad,ad_stat,ADres,p_val_ks,k_stat,KSres,fishers_p_val,odds_ratio,fishers_res])