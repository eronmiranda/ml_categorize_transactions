from utils import import_data, clean_data, prepare_data, create_and_train_model, evaluate_model


# Designed for AMEX Year End Summary format.
def main():
    """Main function to run the transaction categorization."""
    # load and clean data
    # try 'year_end_summaries'
    data = import_data(input("Enter the directory name: "))
    cleaned_data = clean_data(data)
    # prepare data
    x_train, x_test, y_train, y_test = prepare_data(cleaned_data)
    # create a model
    model, vectorizer = create_and_train_model(x_train, y_train)
    # evaluate the model
    print(evaluate_model(model, vectorizer, x_test, y_test))


if __name__ == "__main__":
    main()
