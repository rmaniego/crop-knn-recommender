import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

def main():
    dataset = pd.read_csv("datasets/crops.csv")
    
    # Split the data into features and target variable
    features = dataset.drop("crop", axis=1)
    target = dataset["crop"]
    
    months = list(set(dataset["month"]))
    soil_types = list(set(dataset["soil_type"]))
    soil_drainage = list(set(dataset["soil_drainage"]))
    maturity_in_days = list(set(dataset["maturity_in_days"]))    

    # User input must be appended to the dataset
    # Before preprocessing the features with label encoding
    print("\n[Recommend a Crop]")
    new_features = ["", "", "", 0]
    new_features[0] = ask_for_value(" Select Month: ", months)
    new_features[1] = ask_for_value(" Soil type: ", soil_types)
    new_features[2] = ask_for_value(" Soil drainage: ", soil_drainage)
    new_features[3] = ask_for_value(" Maturity (days): ", maturity_in_days, dtype="int")

    # Create a new DataFrame from the new features list
    new_row = pd.DataFrame([new_features], columns=features.columns)

    # Concatenate the new row with the existing features DataFrame
    features = pd.concat([features, new_row], ignore_index=True)
    
    # Process categorical data into numerical form
    features = features.apply(LabelEncoder().fit_transform)
    
    # Get all except the last row
    features1 = features.drop(features.index[-1])
    features2 = features.tail(1) # This is the user input
    
    # Create and train the KNN classifier
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(features1, target)

    # Predict the probabilities based on the input
    probabilities = knn.predict_proba(features2)[0]
    
    # Map and reverse sort classes with the associated probabilities
    predictions = list(zip(probabilities, knn.classes_))
    predictions = [p for p in predictions if p[0]>0]
    predictions.sort(reverse=True)
    
    # Display the recommended plant for planting
    print("\nRecommendations: ")
    for i, (probability, crop) in enumerate(predictions):
        print(f" {i+1}: {crop}".title())

    # Evaluate the model accuracy
    accuracy = round(knn.score(features1, target)*100)
    print(f"\nAccuracy: {accuracy}%")

def ask_for_value(label, selection, dtype=None):
    selected = ""
    if isinstance(selection[0], int):
        minimum, maximum = min(selection), max(selection)
        print(f"\nRANGE: {minimum}-{maximum} days")
    else:
        options = "\n - ".join(selection)
        print(f"\nOPTIONS:\n - {options}")
    while selected not in selection:
        selected = input(label).strip()
        if dtype == "int":
            selected = int(selected)
    return selected

if __name__ == "__main__":
    main()