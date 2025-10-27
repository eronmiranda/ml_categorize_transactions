from utils import verify_file_exists, import_data, clean_data, prepare_data, create_and_train_model, evaluate_model, save_model, load_model


# Designed for AMEX Year End Summary format.
def main():
    """Main function"""
    if input("Load existing model? (y/n): ").lower() == 'y':
        if not (verify_file_exists('transaction_categorizer.joblib') and
                verify_file_exists('vectorizer.joblib')):
            print("Model or vectorizer file not found. Please train a new model.")
            return
        model, vectorizer = load_model('transaction_categorizer.joblib',
                                       'vectorizer.joblib')
        data = import_data(input("Enter the directory name: "))
        cleaned_data = clean_data(data)
        x = cleaned_data['Category'] + ' ' + cleaned_data['Transaction']
        y = cleaned_data['Sub-Category']
        print(evaluate_model(model, vectorizer, x, y))
    else:
        print("Training a new model...")
        # load and clean data. try input 'year_end_summaries'
        data = import_data(input("Enter the directory name: "))
        cleaned_data = clean_data(data)
        # prepare data
        x_train, x_test, y_train, y_test = prepare_data(cleaned_data)
        # create a model
        model, vectorizer = create_and_train_model(x_train, y_train)
        # save the model and vectorizer
        save_model(model, vectorizer, 'transaction_categorizer.joblib',
                   'vectorizer.joblib')
        # evaluate the model
        print(evaluate_model(model, vectorizer, x_test, y_test))


if __name__ == "__main__":
    main()
