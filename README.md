# Transaction Categorizer

This is a simple Python script that helps categorize credit card transactions from AMEX year-end summary files. I built this to learn about machine learning and to automatically assign categories for my transactions. This will be a preparation for my upcoming personal expense tracker project.

## What it does

- Reads AMEX CSV files and cleans up the transaction data
- Trains a model to predict transaction categories
- Can save the trained model to use again later
- Shows how well the predictions work

## Installation

1. Create and activate a virtual environment:

   ```bash
   python3 -m venv venv
   # On macOS and Linux:
   source venv/bin/activate
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## How to use it

Put your AMEX year-end summary CSV files in the `year_end_summaries` folder, then run:

```bash
python app.py
```

The program will ask if you want to:

- Train a new model (choose 'n') - this takes your CSV files and learns from them
- Use an existing model (choose 'y') - if you've already trained one before

## File structure

```bash
year_end_summaries/     # Put your CSV files here
app.py                  # Main program
utils.py                # Helper functions
results/                # Where outputs go
```

## CSV format

Your AMEX csv files should have these columns:

- Category, Sub-Category, Transaction, Date, etc.

Check the sample file `year_end_summary_20XX_sample.csv` to see the expected format.

## Notes

This was my first machine learning project, so the code is pretty basic. It works by looking at transaction descriptions and trying to predict what sub-category they belong to based on patterns it finds in the training data.

---

**Built with ☕️ by [@eronmiranda](https://github.com/eronmiranda)**
