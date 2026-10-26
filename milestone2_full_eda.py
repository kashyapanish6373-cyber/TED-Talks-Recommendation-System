
"""
TED Talk Recommendation System — Milestone 2: Full EDA
Required input: ted_talks_en.csv in the same folder as this script.

Outputs:
    milestone2_outputs/
        eda_summary.txt
        ted_talks_cleaned_for_recommendation.csv
        plots/
        reports/
"""

from pathlib import Path
import ast
import re
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

BASE_DIR = Path(__file__).resolve().parent
INPUT_FILE = BASE_DIR / "ted_talks_en.csv"
OUT_DIR = BASE_DIR / "milestone2_outputs"
PLOT_DIR = OUT_DIR / "plots"
REPORT_DIR = OUT_DIR / "reports"

for folder in (OUT_DIR, PLOT_DIR, REPORT_DIR):
    folder.mkdir(parents=True, exist_ok=True)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}\n"
        "Place ted_talks_en.csv in the same folder as this script."
    )

df = pd.read_csv(INPUT_FILE, low_memory=False)
original_rows, original_cols = df.shape
summary_lines = []


def log(message=""):
    print(message)
    summary_lines.append(str(message))


def clean_text(value):
    if pd.isna(value):
        return ""
    text = str(value).replace("\u00a0", " ")
    return re.sub(r"\s+", " ", text).strip()


def parse_list(value):
    """Parse list-like values, including Python lists and comma-separated text."""
    if pd.isna(value):
        return []

    if isinstance(value, (list, tuple, set)):
        return [clean_text(x) for x in value if clean_text(x)]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [clean_text(x) for x in parsed if clean_text(x)]
    except (ValueError, SyntaxError):
        pass

    text = text.strip("[]()")
    parts = re.split(r",\s*|;\s*|\|\s*", text)
    return [p.strip(" '\"\t\n") for p in parts if p.strip(" '\"\t\n")]


# Find columns without depending on capitalization or surrounding spaces.
column_lookup = {
    str(c).strip().lower(): c for c in df.columns
}


def col(name):
    return column_lookup.get(name.lower())


def save_bar(series, title, xlabel, ylabel, filename,
             horizontal=False, top_n=15):
    series = series.dropna().head(top_n)

    if series.empty:
        log(f"Skipped chart because there is no data: {title}")
        return

    plt.figure(figsize=(12, 6))

    if horizontal:
        series = series.sort_values()
        series.plot(kind="barh")
    else:
        series.plot(kind="bar")
        plt.xticks(rotation=45, ha="right")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=160)
    plt.close()


def save_histogram(values, title, xlabel, filename):
    values = pd.to_numeric(values, errors="coerce").dropna()

    if values.empty:
        log(f"Skipped histogram because there is no data: {title}")
        return

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=40)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Number of talks")
    plt.tight_layout()
    plt.savefig(PLOT_DIR / filename, dpi=160)
    plt.close()


def save_outliers(data, value_col, output_name, label):
    values = pd.to_numeric(data[value_col], errors="coerce").dropna()

    if values.empty:
        return

    q1, q3 = values.quantile([0.25, 0.75])
    iqr = q3 - q1
    threshold = q3 + 1.5 * iqr
    outliers = data[
        pd.to_numeric(data[value_col], errors="coerce") > threshold
    ]

    id_col = col("talk_id")
    title_col = col("title")
    columns = [
        c for c in [id_col, title_col, value_col] if c is not None
    ]

    outliers[columns].to_csv(REPORT_DIR / output_name, index=False)
    log(f"Potential {label} outliers using IQR rule: {len(outliers)}")


log("=" * 55)
log("TED TALK RECOMMENDATION SYSTEM — FULL EDA")
log("=" * 55)
log(f"Input file: {INPUT_FILE.name}")
log(f"Original rows: {original_rows}")
log(f"Original columns: {original_cols}")
log(f"Dataset shape: {df.shape}")
log("Columns: " + ", ".join(map(str, df.columns)))


# --------------------------------------------------
# 1. DATA QUALITY AND MISSING VALUES
# --------------------------------------------------

log("\n========== 1. DATA QUALITY ==========")

missing = df.isna().sum()
missing_percent = (df.isna().mean() * 100).round(2)

missing_report = pd.DataFrame({
    "missing_count": missing,
    "missing_percent": missing_percent
}).sort_values("missing_count", ascending=False)

missing_report.to_csv(REPORT_DIR / "missing_values.csv")

log(f"Duplicate complete rows: {df.duplicated().sum()}")

id_col = col("talk_id")

if id_col:
    log(f"Duplicate talk IDs: {df[id_col].duplicated().sum()}")

log("Missing values by column:")

for column in missing_report.index:
    count = missing_report.loc[column, "missing_count"]
    percent = missing_report.loc[column, "missing_percent"]

    if count > 0:
        log(f"  {column}: {count} ({percent}%)")

# Empty strings are different from actual NaN values.
empty_values = []

for column in df.columns:
    if (
        pd.api.types.is_object_dtype(df[column])
        or pd.api.types.is_string_dtype(df[column])
    ):
        count = df[column].map(
            lambda x: isinstance(x, str) and not x.strip()
        ).sum()

        empty_values.append({
            "column": column,
            "empty_or_whitespace_count": int(count)
        })

pd.DataFrame(empty_values).to_csv(
    REPORT_DIR / "empty_text_values.csv", index=False
)

plt.figure(figsize=(12, 6))
missing_percent[missing_percent > 0].plot(kind="bar")
plt.title("Missing Values by Column (%)")
plt.xlabel("Column")
plt.ylabel("Missing percentage")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig(PLOT_DIR / "missing_values.png", dpi=160)
plt.close()


# --------------------------------------------------
# 2. TOPIC ANALYSIS
# --------------------------------------------------

log("\n========== 2. TOPIC ANALYSIS ==========")

topic_col = col("topics")
topic_counter = {}
topic_counts_per_talk = []

if topic_col:
    for value in df[topic_col]:
        topics = parse_list(value)
        topic_counts_per_talk.append(len(topics))

        for topic in topics:
            topic_counter[topic] = topic_counter.get(topic, 0) + 1

    topic_series = pd.Series(
        topic_counter, dtype="int64"
    ).sort_values(ascending=False)

    topic_series.rename_axis("topic").reset_index(
        name="talk_count"
    ).to_csv(REPORT_DIR / "topic_counts.csv", index=False)

    if topic_counts_per_talk:
        log(
            "Average topics per talk: "
            f"{np.mean(topic_counts_per_talk):.2f}"
        )

    log("Top 15 topics:")

    for topic, count in topic_series.head(15).items():
        log(f"  {topic}: {count}")

    save_bar(
        topic_series,
        "Top 15 TED Talk Topics",
        "Topic",
        "Number of talks",
        "top_15_topics.png",
        horizontal=True
    )
else:
    log("Topics column not found.")


# --------------------------------------------------
# 3. SPEAKER ANALYSIS
# --------------------------------------------------

log("\n========== 3. SPEAKER ANALYSIS ==========")

speaker1_col = col("speaker_1")
all_speakers_col = col("all_speakers")

speaker_counts = {}
unique_speakers = set()

if speaker1_col:
    for value in df[speaker1_col]:
        name = clean_text(value)

        if name:
            unique_speakers.add(name)
            speaker_counts[name] = speaker_counts.get(name, 0) + 1

if all_speakers_col:
    for value in df[all_speakers_col]:
        for name in parse_list(value):
            if name:
                unique_speakers.add(name)

log(f"Unique speaker names: {len(unique_speakers)}")

if speaker_counts:
    speakers = pd.Series(speaker_counts).sort_values(
        ascending=False
    )

    speakers.rename_axis("speaker").reset_index(
        name="talk_count"
    ).to_csv(REPORT_DIR / "speaker_talk_counts.csv", index=False)

    log("Top 10 primary speakers:")

    for speaker, count in speakers.head(10).items():
        log(f"  {speaker}: {count}")

    save_bar(
        speakers,
        "Top 10 Speakers",
        "Speaker",
        "Number of talks",
        "top_10_speakers.png",
        horizontal=True,
        top_n=10
    )


# --------------------------------------------------
# 4. TEXT ANALYSIS
# --------------------------------------------------

log("\n========== 4. TEXT ANALYSIS ==========")

text_fields = [
    "title",
    "description",
    "transcript",
    "about_speakers"
]

text_review = []

for field in text_fields:
    column = col(field)

    if not column:
        log(f"{field}: column not found.")
        continue

    values = df[column].map(clean_text)
    word_counts = values.map(
        lambda text: len(text.split())
    )

    log(
        f"{field}: empty={(values == '').sum()}, "
        f"mean words={word_counts.mean():.1f}, "
        f"median words={word_counts.median():.1f}, "
        f"minimum={word_counts.min()}, "
        f"maximum={word_counts.max()}"
    )

    if field == "title":
        short_limit, long_limit = 3, 25
    elif field == "description":
        short_limit, long_limit = 20, 500
    elif field == "transcript":
        short_limit, long_limit = 100, 20000
    else:
        short_limit, long_limit = 5, 500

    for index, (text, count) in enumerate(
        zip(values, word_counts)
    ):
        flag = ""

        if count == 0:
            flag = "empty"
        elif count < short_limit:
            flag = "unusually_short_review"
        elif count > long_limit:
            flag = "unusually_long_review"

        if flag:
            text_review.append({
                "row_index": index,
                "talk_id": (
                    df.iloc[index][id_col] if id_col else ""
                ),
                "field": field,
                "word_count": int(count),
                "flag": flag,
                "text_preview": text[:240]
            })

    if field == "transcript":
        log(
            "Transcripts under 100 words: "
            f"{(word_counts < 100).sum()}"
        )
        log(
            "Transcripts over 20,000 words: "
            f"{(word_counts > 20000).sum()}"
        )

pd.DataFrame(text_review).to_csv(
    REPORT_DIR / "text_entries_for_review.csv", index=False
)


# --------------------------------------------------
# 5. VIEWS ANALYSIS
# --------------------------------------------------

log("\n========== 5. VIEWS ANALYSIS ==========")

views_col = col("views")

if views_col:
    df[views_col] = pd.to_numeric(
        df[views_col], errors="coerce"
    )

    views = df[views_col].dropna()

    if not views.empty:
        log(
            f"Mean views: {views.mean():,.2f}\n"
            f"Median views: {views.median():,.2f}\n"
            f"Minimum views: {views.min():,.0f}\n"
            f"Maximum views: {views.max():,.0f}"
        )

        selected_cols = [
            c for c in [
                id_col, col("title"), speaker1_col, views_col
            ] if c is not None
        ]

        df.nlargest(10, views_col)[selected_cols].to_csv(
            REPORT_DIR / "top_10_viewed_talks.csv",
            index=False
        )

        save_histogram(
            views,
            "Distribution of TED Talk Views",
            "Views",
            "views_histogram.png"
        )

        save_histogram(
            np.log1p(views),
            "Log-Transformed Distribution of Views",
            "log(views + 1)",
            "views_log_distribution.png"
        )

        save_outliers(
            df,
            views_col,
            "views_outliers_iqr.csv",
            "high-view"
        )


# --------------------------------------------------
# 6. DURATION ANALYSIS
# --------------------------------------------------

log("\n========== 6. DURATION ANALYSIS ==========")

duration_col = col("duration")

if duration_col:
    df[duration_col] = pd.to_numeric(
        df[duration_col], errors="coerce"
    )

    durations = df[duration_col].dropna()

    log(
        "Missing or non-numeric duration: "
        f"{df[duration_col].isna().sum()}"
    )

    log(
        "Zero or negative duration: "
        f"{(df[duration_col] <= 0).sum()}"
    )

    if not durations.empty:
        log(
            f"Mean duration: {durations.mean():.2f} seconds "
            f"({durations.mean() / 60:.2f} minutes)"
        )
        log(
            f"Median duration: {durations.median():.2f} seconds "
            f"({durations.median() / 60:.2f} minutes)"
        )
        log(f"Minimum duration: {durations.min():.0f} seconds")
        log(f"Maximum duration: {durations.max():.0f} seconds")

        save_histogram(
            durations,
            "Distribution of TED Talk Duration",
            "Duration (seconds)",
            "duration_histogram.png"
        )

        save_outliers(
            df,
            duration_col,
            "duration_outliers_iqr.csv",
            "long-duration"
        )


# --------------------------------------------------
# 7. DATE AND YEARLY TREND ANALYSIS
# --------------------------------------------------

log("\n========== 7. DATE ANALYSIS ==========")

recorded_col = col("recorded_date")
published_col = col("published_date")

for field, column in [
    ("recorded_date", recorded_col),
    ("published_date", published_col)
]:
    if not column:
        log(f"{field}: column not found.")
        continue

    dates = pd.to_datetime(
        df[column], errors="coerce", utc=True
    )

    log(
        f"{field}: missing or invalid dates = "
        f"{dates.isna().sum()}"
    )

    if field == "published_date":
        valid_dates = dates.dropna()

        if not valid_dates.empty:
            log(f"Earliest publication: {valid_dates.min().date()}")
            log(f"Latest publication: {valid_dates.max().date()}")

            yearly = valid_dates.dt.year.value_counts().sort_index()

            yearly.rename_axis("year").reset_index(
                name="talk_count"
            ).to_csv(
                REPORT_DIR / "talks_published_per_year.csv",
                index=False
            )

            plt.figure(figsize=(12, 6))
            yearly.plot(kind="line", marker="o")
            plt.title("TED Talks by Publication Year")
            plt.xlabel("Publication year")
            plt.ylabel("Number of talks")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(
                PLOT_DIR / "yearly_publication_trend.png",
                dpi=160
            )
            plt.close()


# --------------------------------------------------
# 8. LANGUAGE ANALYSIS
# --------------------------------------------------

log("\n========== 8. LANGUAGE ANALYSIS ==========")

native_col = col("native_lang")
available_col = col("available_lang")

if native_col:
    native_counts = (
        df[native_col]
        .fillna("Missing")
        .astype(str)
        .str.strip()
        .replace("", "Missing")
        .value_counts()
    )

    native_counts.rename_axis(
        "native_language"
    ).reset_index(
        name="talk_count"
    ).to_csv(
        REPORT_DIR / "native_language_counts.csv",
        index=False
    )

    log("Top 10 native languages:")

    for language, count in native_counts.head(10).items():
        log(f"  {language}: {count}")

    save_bar(
        native_counts,
        "Top Native Languages",
        "Language",
        "Number of talks",
        "native_languages.png",
        horizontal=True,
        top_n=10
    )

if available_col:
    available_counts = {}

    for value in df[available_col]:
        for language in parse_list(value):
            available_counts[language] = (
                available_counts.get(language, 0) + 1
            )

    available_series = pd.Series(
        available_counts, dtype="int64"
    ).sort_values(ascending=False)

    available_series.rename_axis(
        "language"
    ).reset_index(
        name="talk_count_available"
    ).to_csv(
        REPORT_DIR / "available_language_counts.csv",
        index=False
    )

    log(
        "Distinct available language labels: "
        f"{len(available_counts)}"
    )
    log(
        "Missing available-language values: "
        f"{df[available_col].isna().sum()}"
    )


# --------------------------------------------------
# 9. EVENTS AND OCCUPATIONS
# --------------------------------------------------

log("\n========== 9. EVENTS AND OCCUPATIONS ==========")

event_col = col("event")
occupation_col = col("occupations")

if event_col:
    event_counts = (
        df[event_col]
        .fillna("")
        .astype(str)
        .str.strip()
        .replace("", np.nan)
        .value_counts()
    )

    event_counts.rename_axis("event").reset_index(
        name="talk_count"
    ).to_csv(REPORT_DIR / "event_counts.csv", index=False)

    log("Top 10 events:")

    for event, count in event_counts.head(10).items():
        log(f"  {event}: {count}")

    save_bar(
        event_counts,
        "Top 10 TED Talk Events",
        "Event",
        "Number of talks",
        "top_10_events.png",
        horizontal=True,
        top_n=10
    )

if occupation_col:
    occupation_counts = {}

    for value in df[occupation_col]:
        for occupation in parse_list(value):
            occupation_counts[occupation] = (
                occupation_counts.get(occupation, 0) + 1
            )

    occupation_series = pd.Series(
        occupation_counts, dtype="int64"
    ).sort_values(ascending=False)

    occupation_series.rename_axis(
        "occupation"
    ).reset_index(
        name="talk_count"
    ).to_csv(
        REPORT_DIR / "occupation_counts.csv",
        index=False
    )

    log("Top 15 occupations:")

    for occupation, count in occupation_series.head(15).items():
        log(f"  {occupation}: {count}")

    save_bar(
        occupation_series,
        "Top 15 Speaker Occupations",
        "Occupation",
        "Number of talks",
        "top_15_occupations.png",
        horizontal=True,
        top_n=15
    )


# --------------------------------------------------
# 10. COMMENTS ANALYSIS
# --------------------------------------------------

log("\n========== 10. COMMENTS ANALYSIS ==========")

comments_col = col("comments")

if comments_col:
    df[comments_col] = pd.to_numeric(
        df[comments_col], errors="coerce"
    )

    comments = df[comments_col].dropna()

    log(
        f"Missing comments: {df[comments_col].isna().sum()}"
    )

    if not comments.empty:
        log(f"Mean comments: {comments.mean():.2f}")
        log(f"Median comments: {comments.median():.2f}")
        log(f"Minimum comments: {comments.min():.0f}")
        log(f"Maximum comments: {comments.max():.0f}")

        selected_cols = [
            c for c in [id_col, col("title"), comments_col]
            if c is not None
        ]

        df.nlargest(10, comments_col)[selected_cols].to_csv(
            REPORT_DIR / "top_10_commented_talks.csv",
            index=False
        )

        save_histogram(
            comments,
            "Distribution of TED Talk Comments",
            "Comments",
            "comments_histogram.png"
        )

        save_outliers(
            df,
            comments_col,
            "comments_outliers_iqr.csv",
            "high-comment"
        )


# --------------------------------------------------
# 11. RELATED TALK REFERENCE VALIDATION
# --------------------------------------------------

log("\n========== 11. RELATED TALKS ANALYSIS ==========")

related_col = col("related_talks")

if related_col and id_col:
    known_ids = set(
        pd.to_numeric(df[id_col], errors="coerce")
        .dropna()
        .astype(int)
    )

    related_rows = []

    for _, row in df.iterrows():
        references = parse_list(row[related_col])
        found = 0
        not_found = 0

        for reference in references:
            match = re.search(r"\d+", str(reference))

            if match:
                reference_id = int(match.group())

                if reference_id in known_ids:
                    found += 1
                else:
                    not_found += 1
            else:
                not_found += 1

        related_rows.append({
            "talk_id": row[id_col],
            "related_talk_count": len(references),
            "references_found_in_dataset": found,
            "references_not_found_or_unparsed": not_found
        })

    related_report = pd.DataFrame(related_rows)

    related_report.to_csv(
        REPORT_DIR / "related_talks_validation.csv",
        index=False
    )

    log(
        "Average related talks per entry: "
        f"{related_report['related_talk_count'].mean():.2f}"
    )

    log(
        "Related references not found or unparsed: "
        f"{related_report['references_not_found_or_unparsed'].sum()}"
    )

    log(
        "Related-talk references may help with offline checks, "
        "but do not prove personalized relevance."
    )
else:
    log("Related-talk validation skipped: required columns missing.")


# --------------------------------------------------
# 12. DATA CLEANING AND RECOMMENDATION PREPARATION
# --------------------------------------------------

log("\n========== 12. CLEANING AND PREPARATION ==========")

clean = df.copy()

before = len(clean)
clean = clean.drop_duplicates()
log(f"Duplicate complete rows removed: {before - len(clean)}")

if id_col:
    before = len(clean)
    clean = clean.drop_duplicates(
        subset=[id_col], keep="first"
    )
    log(f"Duplicate talk IDs removed: {before - len(clean)}")

# Standardize whitespace in text columns.
for column in clean.columns:
    if (
        pd.api.types.is_object_dtype(clean[column])
        or pd.api.types.is_string_dtype(clean[column])
    ):
        clean[column] = clean[column].map(clean_text)

# Fill optional metadata with empty strings, not invented data.
for field in [
    "all_speakers",
    "occupations",
    "about_speakers",
    "available_lang",
    "related_talks"
]:
    column = col(field)

    if column:
        clean[column] = clean[column].fillna("")

# Convert date fields.
for column in [recorded_col, published_col]:
    if column:
        clean[column] = pd.to_datetime(
            clean[column], errors="coerce", utc=True
        )

title_col = col("title")
description_col = col("description")
transcript_col = col("transcript")

# A missing title is not useful as a recommendation item.
if title_col:
    clean = clean[
        clean[title_col].fillna("").astype(str).str.strip().ne("")
    ]

content_cols = [
    column for column in [
        title_col, topic_col, description_col, transcript_col
    ]
    if column is not None
]

if content_cols:
    for column in content_cols:
        clean[column] = clean[column].fillna("").astype(str)

    clean["combined_text"] = clean[content_cols].agg(
        " ".join, axis=1
    ).map(lambda text: re.sub(r"\s+", " ", text).strip())

    clean = clean[clean["combined_text"].str.len() > 0]

    clean["combined_text_word_count"] = clean[
        "combined_text"
    ].map(lambda text: len(text.split()))

clean_path = OUT_DIR / "ted_talks_cleaned_for_recommendation.csv"
clean.to_csv(clean_path, index=False)

log(f"Cleaned dataset saved: {clean_path}")
log(f"Final dataset shape: {clean.shape}")
log(
    "Net rows removed, including duplicates and unusable titles/content: "
    f"{original_rows - len(clean)}"
)


# --------------------------------------------------
# 13. FINAL FINDINGS AND READINESS
# --------------------------------------------------

log("\n========== 13. FINAL FINDINGS ==========")

log(
    "Content-based features: title, topics, description, transcript."
)
log(
    "Possible metadata features: speaker, occupations, event, "
    "duration, views, and native language."
)
log(
    "Views and comments represent popularity or engagement, "
    "not an individual user's personal preferences."
)
log(
    "Short-text flags are candidates for review, not proof of bad data."
)
log(
    "Recommendation quality must be evaluated separately; "
    "cleaning and EDA alone do not prove recommendation accuracy."
)
log(f"Charts folder: {PLOT_DIR}")
log(f"Reports folder: {REPORT_DIR}")

summary_path = OUT_DIR / "eda_summary.txt"
summary_path.write_text(
    "\n".join(summary_lines), encoding="utf-8"
)

print("\nEDA completed successfully!")
print(f"Summary: {summary_path}")
print(f"Output folder: {OUT_DIR}")
