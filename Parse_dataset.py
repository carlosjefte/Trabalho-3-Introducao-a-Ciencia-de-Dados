import pandas as pd

# Load the dataset
file_path = 'Crânios Egípcios.csv'
df_cleaned = pd.read_csv(file_path, skiprows=1)

# Rename columns based on the content from the header rows
df_cleaned.columns = ['Crânios', 'Primitivo_X1', 'Primitivo_X2', 'Primitivo_X3', 'Primitivo_X4',
                      'Antigo_X1', 'Antigo_X2', 'Antigo_X3', 'Antigo_X4',
                      'Dinastias_X1', 'Dinastias_X2', 'Dinastias_X3', 'Dinastias_X4',
                      'Ptolemaico_X1', 'Ptolemaico_X2', 'Ptolemaico_X3', 'Ptolemaico_X4',
                      'Romano_X1', 'Romano_X2', 'Romano_X3', 'Romano_X4']

# Initialize lists to hold the data for the new DataFrame
X1 = []
X2 = []
X3 = []
X4 = []
labels = []

# Define the period map
period_map = {
  'Primitivo': 'Pré-dinástico primitivo',
  'Antigo': 'Pré-dinástico antigo',
  'Dinastias': '12 e 13 dinastias',
  'Ptolemaico': 'Período ptolemaico',
  'Romano': 'Período romano'
}

# Iterate through each row in the dataframe
for idx in range(len(df_cleaned)):
  sample = df_cleaned.iloc[idx]

  # Iterate through each column and assign values based on the period
  for period, label in period_map.items():
    for col in range(1, 5):  # Iterate through X1 to X4
      col_name = f'{period}_X{col}'
      if col_name in sample and not pd.isnull(sample[col_name]):
        if col == 1:
          X1.append(sample[col_name])
        elif col == 2:
          X2.append(sample[col_name])
        elif col == 3:
          X3.append(sample[col_name])
        elif col == 4:
          X4.append(sample[col_name])
    labels.append(label)

# Check that all arrays have the same length
print(f"X1: {len(X1)}\nX2: {len(X2)}\nX3: {len(X3)}\nX4: {len(X4)}\nLabels: {len(labels)}")

# Create a DataFrame with the selected columns
df_final = pd.DataFrame({
  'X1': X1,
  'X2': X2,
  'X3': X3,
  'X4': X4,
  'label': labels
})

# Display the first few rows to check the result
print(df_final.head())

# Save the formatted dataset to a new CSV file
df_final.to_csv('Crânios Egípcios Formatado.csv', index=False)