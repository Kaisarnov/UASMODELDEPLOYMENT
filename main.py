from pipeline import CreditScorePipeline


def main():

    pipeline = CreditScorePipeline()

    pipeline.run("data_C.csv")


if __name__ == "__main__":
    main()