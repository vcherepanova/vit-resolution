import os
import glob
import pandas as pd
os.chdir("results_interpolation_inference")

extension = 'csv'
all_filenames = [i for i in glob.glob('*.{}'.format(extension))]
#combine all files in the list
combined_csv = pd.concat([pd.read_csv(f) for f in all_filenames ])
#export to csv
print(combined_csv)
print(len(all_filenames), combined_csv.shape)
combined_csv['filename'] = all_filenames
combined_csv.to_csv( "combined_results.csv", index=False, encoding='utf-8-sig')