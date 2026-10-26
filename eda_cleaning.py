
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ast

# =====================================
# 1. LOAD DATASET
# =====================================
df = pd.read_csv("ted_talks_en.csv")

print("Original dataset shape:", df.shape)

# =====================================
# 2. INITIAL EDA
# =====================================
print("\nFirst five rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nDuplicate rows:", df.duplicated().sum())
print("Duplicate talk IDs:", df["talk_id"].duplicated().sum())

# =====================================
# 3. DATA CLEANING
# =====================================
clean_df = df.copy()

# Convert dates
for col in ["recorded_date", "published_date"]:
    clean_df[col] = pd.to_datetime(
        clean_df[col], errors="coerce"
    )

# Fill missing optional speaker metadata
for col in ["all_speakers", "occupations", "about_speakers"]:
    clean_df[col] = clean_df[col].fillna("")

# Keep missing comments as missing, not zero.
# Remove records only if essential text is missing.
clean_df = clean_df.dropna(
    subset=["title", "transcript"]
)

# Remove blank titles or transcripts
for col in ["title", "transcript"]:
    clean_df = clean_df[
        clean_df[col].astype(str).str.strip().ne("")
    ]

print("\nMissing or invalid dates after conversion:")
print(clean_df[["recorded_date", "published_date"]].isnull().sum())

print("\nCleaned dataset shape:", clean_df.shape)

print("\nMissing values after cleaning:")
print(clean_df.isnull().sum())

# Save cleaned dataset
clean_df.to_csv("ted_talks_cleaned.csv", index=False)
print("\nCleaned dataset saved!")

# =====================================
# 4. PREPARE OUTPUT FOLDER
# =====================================
import os
os.makedirs("eda_plots", exist_ok=True)

sns.set_theme(style="whitegrid")

# =====================================
# 5. DISTRIBUTION OF VIEWS
# =====================================
plt.figure(figsize=(9, 5))
sns.histplot(clean_df["views"], bins=40)
plt.title("Distribution of TED Talk Views")
plt.xlabel("Views")
plt.ylabel("Number of Talks")
plt.tight_layout()
plt.savefig("eda_plots/views_distribution.png")
plt.show()

# Views are often highly skewed.
# Log scale helps reveal the distribution.
plt.figure(figsize=(9, 5))
sns.histplot(
    clean_df.loc[clean_df["views"] > 0, "views"].apply(
        lambda x: __import__("math").log10(x)
    ),
    bins=40
)
plt.title("Distribution of TED Talk Views (Log10 Scale)")
plt.xlabel("Log10 of Views")
plt.ylabel("Number of Talks")
plt.tight_layout()
plt.savefig("eda_plots/views_log_distribution.png")
plt.show()

# =====================================
# 6. DISTRIBUTION OF DURATION
# =====================================
plt.figure(figsize=(9, 5))
sns.histplot(clean_df["duration"] / 60, bins=40)
plt.title("Distribution of TED Talk Duration")
plt.xlabel("Duration (minutes)")
plt.ylabel("Number of Talks")
plt.tight_layout()
plt.savefig("eda_plots/duration_distribution.png")
plt.show()

# =====================================
# 7. TOP 10 MOST-VIEWED TALKS
# =====================================
top_talks = clean_df.nlargest(10, "views")

plt.figure(figsize=(11, 7))
sns.barplot(
    data=top_talks,
    x="views",
    y="title"
)
plt.title("Top 10 Most-Viewed TED Talks")
plt.xlabel("Views")
plt.ylabel("Talk title")
plt.tight_layout()
plt.savefig("eda_plots/top_10_talks.png")
plt.show()

# =====================================
# 8. MOST COMMON TOPICS
# =====================================
def parse_topics(value):
    if isinstance(value, list):
        return value
    if pd.isna(value):
        return []
    try:
        result = ast.literal_eval(str(value))
        return result if isinstance(result, list) else []
    except (ValueError, SyntaxError):
        return []

if "topics" in clean_df.columns:
    topic_lists = clean_df["topics"].apply(parse_topics)
    all_topics = topic_lists.explode()

    all_topics = all_topics[
        all_topics.notna() & all_topics.ne("")
    ]

    top_topics = all_topics.value_counts().head(10)

    if not top_topics.empty:
        plt.figure(figsize=(10, 6))
        sns.barplot(
            x=top_topics.values,
            y=top_topics.index
        )
        plt.title("Top 10 Most Common TED Talk Topics")
        plt.xlabel("Number of Talks")
        plt.ylabel("Topic")
        plt.tight_layout()
        plt.savefig("eda_plots/top_10_topics.png")
        plt.show()

# =====================================
# 9. MISSING VALUES VISUALIZATION
# =====================================
missing = clean_df.isnull().sum()
missing = missing[missing > 0].sort_values(ascending=False)

if not missing.empty:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=missing.values, y=missing.index)
    plt.title("Missing Values by Column")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Column")
    plt.tight_layout()
    plt.savefig("eda_plots/missing_values.png")
    plt.show()
else:
    print("No missing values remain.")

# =====================================
# 10. FINAL SUMMARY
# =====================================
print("\nEDA COMPLETED")
print("Original rows:", len(df))
print("Cleaned rows:", len(clean_df))
print("Columns:", len(clean_df.columns))
print("Plots saved in the eda_plots folder.")
