import pandas as pd

data = pd.read_csv("/content/reddit_cohere.csv")

machine_df = data[['machine_text']].dropna().copy()
machine_df['labels'] = 0  # Label '0' for machine-generated text

# Step 2: Create a DataFrame for 'human_text'
human_df = data[['human_text']].copy()
human_df['labels'] = 1  # Label '1' for human-generated text

# Step 3: Rename the columns to have consistent names
machine_df.rename(columns={'machine_text': 'text'}, inplace=True)
human_df.rename(columns={'human_text': 'text'}, inplace=True)

# Step 4: Concatenate the two DataFrames to form a single DataFrame
new_df = pd.concat([machine_df, human_df], ignore_index=True)


# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-classification", model="TrustSafeAI/RADAR-Vicuna-7B")



def get_prediction(text):
  try:
    prediction = pipe(text)
    return prediction[0]['label']
  except:
    return None


new_df['prediction'] = new_df['text'].apply(get_prediction)

new_df.to_csv("merged_data_predictions.csv", index=False)

# removing None values
df_cleaned = new_df.dropna(subset=['prediction'])

df_cleaned['prediction'] = df_cleaned['prediction'].map({'LABEL_1': 1, 'LABEL_0': 0})

actual_labls = df_cleaned["labels"].tolist()
predicted_labels = df_cleaned["prediction"].tolist()

from sklearn.metrics import precision_recall_fscore_support
prf = precision_recall_fscore_support(actual_labls, predicted_labels, average='macro')

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(actual_labls, predicted_labels)

from sklearn.metrics import classification_report
target_names = ['machine_generated', 'human_written']
cr = classification_report(actual_labls, predicted_labels, target_names=target_names)

from sklearn.metrics import roc_auc_score
auc = roc_auc_score(actual_labls, predicted_labels)

# Save the report to a text file
with open("classification_report.txt", "w") as file:
    file.write(cr)
    file.write("\n")
    file.write(cm)
    file.write("\n")
    file.write(prf)
    file.write("\n")
    file.write(auc)


print("Classification report saved to classification_report.txt")



