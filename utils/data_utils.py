from sklearn.model_selection import train_test_split


def filter_classes(df, top_n=10):
    print("\n" + "***** FILTERING CLASSES *****")
    counts = df["medical_specialty"].value_counts()

    valid_classes = counts.head(top_n).index
    filtered_df = df[df["medical_specialty"].isin(valid_classes)]

    print(f"Keeping top {top_n} classes")
    print(f"Resultant length of dataframe: {len(filtered_df)}")

    return filtered_df


def remove_empty_transcription(df):
    print("\n" + "***** REMOVING EMPTY ENTRIES *****")
    df = df.copy()
    df["text"] = df["transcription"].fillna("")

    df = df[df["text"].str.len() > 50]

    print("Removed entries with empty transcription!")
    print(f"{len(df)} entries after filtering")

    return df


def encode_labels(df):
    print("\n" + "***** ADDING ENCODINGS FOR LABELS *****")
    medical_specialties = sorted(df["medical_specialty"].unique())
    label_encodings = {name: index for index, name in enumerate(medical_specialties)}

    df = df.copy()
    df["label"] = df["medical_specialty"].map(label_encodings)

    print("Labels encoded for data!")
    return df, label_encodings


def split_data(df, test_size=0.2, valid_size=0.1, seed=42):
    print("\n" + "***** SPLITTING DATA *****")
    train_valid, test = train_test_split(
        df, test_size=test_size, stratify=df["label"], random_state=seed
    )

    valid_ratio = valid_size / (1 - test_size)
    train, valid = train_test_split(
        train_valid,
        test_size=valid_ratio,
        stratify=train_valid["label"],
        random_state=seed,
    )

    print(f"Length of Train set: {len(train)}")
    print(f"Length of Valid set: {len(valid)}")
    print(f"Length of Test set: {len(test)}")

    return train, valid, test
