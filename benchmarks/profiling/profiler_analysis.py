import pandas as pd
from time_profiler import get_profiler

pd.set_option("display.max_colwidth", None)

profiler = get_profiler()


def main():
    df = profiler.read_events_from_file_csv()

    epoch_totals = df.groupby(["name", "epoch"], as_index=False).agg(
        total_time=("time", "sum"), total_calls=("time", "count")
    )

    mean_total_per_name = epoch_totals.groupby("name", as_index=False).agg(
        mean_total_time=("total_time", "mean"), total_calls=("total_calls", "sum")
    )

    mean_total_per_name = mean_total_per_name.sort_values(
        by="mean_total_time", ascending=False
    )

    print(mean_total_per_name)

    mean_total_per_name.to_csv("mean_total_per_name.csv", index=False)


if __name__ == "__main__":
    main()
