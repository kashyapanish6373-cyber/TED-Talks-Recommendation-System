
import pandas as pd
from pathlib import Path

# 1. Load the original dataset
input_file = Path("ted_talks_en.csv")
df = pd.read_csv(input_file)

print("========== DATASET AUDIT ==========")
print("Dataset source:", input_file)
print("Original shape:", df.shape)

# 2. Inspect available columns
print("\nAvailable columns:")
for column in df.columns:
    print(f"- {column}: {df[column].dtype}")

# 3. Check missing and empty values
print("\nMissing values before cleaning:")
print(df.isnull().sum())

print("\nEmpty or whitespace-only text values:")
for column in df.select_dtypes(include="object").columns:
    empty_count = (
        df[column].fillna("").astype(str).str.strip().eq("").sum()
    )
    if empty_count > 0:
        print(f"{column}: {empty_count}")

# 4. Check duplicates
print("\nDuplicate full rows:", df.duplicated().sum())
print("Duplicate talk IDs:", df["talk_id"].duplicated().sum())

# 5. Check important recommendation fields
required_fields = [
    "talk_id", "title", "speaker_1",
    "description", "topics", "transcript", "url"
]

print("\nRequired fields:")
for column in required_fields:
    if column in df.columns:
        print(f"{column}: available")
    else:
        print(f"{column}: MISSING")

# 6. Check for user-behaviour data
user_related_fields = [
    column for column in df.columns
    if any(word in column.lower() for word in [
        "user_id", "rating", "watch_history",
        "viewing_history", "preference"
    ])
]

print("\nPotential user-behaviour columns:")
print(user_related_fields if user_related_fields else
      "None found by column-name check")

# 7. Clean duplicate records
clean_df = df.drop_duplicates().copy()
clean_df = clean_df.drop_duplicates(
    subset=["talk_id"], keep="first"
).copy()

# 8. Standardize text fields
text_columns = [
    "title", "speaker_1", "description",
    "topics", "transcript", "url",
    "all_speakers", "occupations",
    "about_speakers", "event"
]

for column in text_columns:
    if column in clean_df.columns:
        clean_df[column] = clean_df[column].map(
            lambda value: value.strip()
            if isinstance(value, str) else value
        )

# Convert blank text into missing values for validation
for column in [
    "title", "description", "topics",
    "transcript", "url"
]:
    if column in clean_df.columns:
        clean_df[column] = clean_df[column].replace(
            r"^\s*$", pd.NA, regex=True
        )

# A talk needs a title and at least one content field.
content_columns = [
    column for column in ["description", "topics", "transcript"]
    if column in clean_df.columns
]

clean_df = clean_df.dropna(subset=["title"])

if content_columns:
    clean_df = clean_df.dropna(
        subset=content_columns, how="all"
    )

# 9. Keep fields useful for recommendations
useful_columns = [
    "talk_id", "title", "speaker_1", "all_speakers",
    "description", "topics", "transcript", "url",
    "views", "duration", "event",
    "recorded_date", "published_date", "native_lang"
]

available_columns = [
    column for column in useful_columns
    if column in clean_df.columns
]

recommendation_df = clean_df[available_columns].copy()

# 10. Convert date fields
for column in ["recorded_date", "published_date"]:
    if column in recommendation_df.columns:
        recommendation_df[column] = pd.to_datetime(
            recommendation_df[column], errors="coerce"
        )

# 11. Save the recommendation-ready dataset
output_file = Path("ted_talks_recommendation_ready.csv")
recommendation_df.to_csv(output_file, index=False)

# 12. Verify the saved file
saved_df = pd.read_csv(output_file)

print("\n========== FINAL RESULTS ==========")
print("Original shape:", df.shape)
print("Prepared shape:", recommendation_df.shape)
print("Saved file:", output_file)
print("Saved file verified:", output_file.exists())
print("Duplicate rows remaining:", saved_df.duplicated().sum())
print("Duplicate talk IDs remaining:",
      saved_df["talk_id"].duplicated().sum())
print("\nMissing values in prepared dataset:")
print(saved_df.isnull().sum())
print("\nMilestone 1 dataset preparation completed.")

import pandas as pd
from pathlib import Path

# Load original dataset
df = pd.read_csv("ted_talks_en.csv")

print("========== DATASET AUDIT ==========")
print("Original dataset shape:", df.shape)

print("\nColumns and data types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isnull().sum())

print("\nEmpty descriptions:")
if "description" in df.columns:
    empty_descriptions = (
        df["description"].fillna("").astype(str).str.strip().eq("").sum()
    )
    print(empty_descriptions)

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate talk IDs:", df["talk_id"].duplicated().sum())

# Check whether user-behaviour data is available
user_fields = [
    col for col in df.columns
    if any(word in col.lower() for word in
           ["user_id", "rating", "watch_history",
            "viewing_history", "preference"])
]
print("\nPotential user-behaviour columns:", user_fields or "None found")

# Remove duplicate rows and duplicate talk IDs
clean_df = df.drop_duplicates().copy()
clean_df = clean_df.drop_duplicates(
    subset="talk_id", keep="first"
).copy()

# Clean whitespace in text fields
for col in clean_df.select_dtypes(include="object").columns:
    clean_df[col] = clean_df[col].map(
        lambda value: value.strip() if isinstance(value, str) else value
    )

# Treat blank strings in important fields as missing
for col in ["title", "description", "topics", "transcript", "url"]:
    if col in clean_df.columns:
        clean_df[col] = clean_df[col].replace(r"^\s*$", pd.NA, regex=True)

# Keep talks with a title and at least one content field
content_cols = [
    col for col in ["description", "topics", "transcript"]
    if col in clean_df.columns
]

clean_df = clean_df.dropna(subset=["title"])

if content_cols:
    clean_df = clean_df.dropna(subset=content_cols, how="all")

# Select useful recommendation fields
useful_cols = [
    "talk_id", "title", "speaker_1", "all_speakers",
    "description", "topics", "transcript", "url",
    "views", "duration", "event", "recorded_date",
    "published_date", "native_lang"
]
selected_cols = [col for col in useful_cols if col in clean_df.columns]
recommendation_df = clean_df[selected_cols].copy()

# Convert date fields
for col in ["recorded_date", "published_date"]:
    if col in recommendation_df.columns:
        recommendation_df[col] = pd.to_datetime(
            recommendation_df[col], errors="coerce"
        )

# Save prepared dataset
output_file = Path("ted_talks_recommendation_ready.csv")
recommendation_df.to_csv(output_file, index=False)

# Verify saved file
saved_df = pd.read_csv(output_file)

print("\n========== FINAL RESULTS ==========")
print("Original shape:", df.shape)
print("Prepared shape:", saved_df.shape)
print("Saved file:", output_file.resolve())
print("Duplicate rows remaining:", saved_df.duplicated().sum())
print(
    "Duplicate talk IDs remaining:",
    saved_df["talk_id"].duplicated().sum()
)
print("\nMissing values in prepared dataset:")
print(saved_df.isnull().sum())

print("\nMilestone 1 preparation completed!")
# Fill missing optional speaker metadata
for col in ["all_speakers", "occupations", "about_speakers"]:
    if col in clean_df.columns:
        clean_df[col] = clean_df[col].fillna("")