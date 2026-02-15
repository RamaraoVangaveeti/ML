import pandas as pd
from sklearn.model_selection import train_test_split

INPUT = "Pokemon.csv"
TRAIN_OUT = "Train_Data.csv"
TEST_OUT = "Test_Data.csv"

if __name__ == "__main__":
    df = pd.read_csv(INPUT)
    if 'Legendary' not in df.columns:
        raise SystemExit('Legendary column not found in dataset')
    # drop obvious id columns if present
    df = df.drop(columns=["#", "Id", "ID", "index"], errors='ignore')
    # keep original order
    df = df.reset_index(drop=True)
    # stratify on Legendary if possible
    y = df['Legendary']
    # ensure boolean or categorical
    try:
        y = y.astype(bool)
    except Exception:
        y = pd.Categorical(y)
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=y)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"Saved {TRAIN_OUT} ({len(train)}) and {TEST_OUT} ({len(test)})")
