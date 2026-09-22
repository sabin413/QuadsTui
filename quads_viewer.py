from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
from dateutil.relativedelta import relativedelta
from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.widgets import DataTable, Footer, Header, Static
from textual_plotext import PlotextPlot


FILE_RE = re.compile(
    r"^quads_results_(?P<model>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_summary\.pkl$"
)
PARQUET_RE = re.compile(
    r"^quads_results_(?P<model>.+?)_(?P<date>\d{4}_\d{2}_\d{2})_violations_raw_data\.parquet$"
)


def shorten(value: Any, width: int = 60) -> str:
    text = str(value)
    if len(text) <= width:
        return text
    return text[: width - 1] + "…"


def historical_reference_window(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    start = dt - relativedelta(years=1, months=1)
    end = dt - relativedelta(years=1) + relativedelta(months=1)
    return f"{start:%Y-%m} to {end:%Y-%m}"


def to_1d_float_list(value: Any) -> list[float]:
    if value is None:
        return []

    if isinstance(value, list):
        seq = value
    elif isinstance(value, tuple):
        seq = list(value)
    elif hasattr(value, "tolist"):
        out = value.tolist()
        seq = out if isinstance(out, list) else [out]
    else:
        try:
            seq = list(value)
        except TypeError:
            return []

    result: list[float] = []
    for item in seq:
        try:
            result.append(float(item))
        except Exception:
            return []
    return result


class QuantilePlot(PlotextPlot):
    def on_mount(self) -> None:
        self.plt.title("Quantile plot")
        self.plt.grid(True, True)

    def show_message(self, title: str) -> None:
        self.plt.clear_data()
        self.plt.clear_figure()
        self.plt.title(title)
        self.refresh()

    def show_row_plot(
        self,
        x_vals: list[float],
        y_vals: list[float],
        title: str,
        fence_low: float | None = None,
        fence_high: float | None = None,
    ) -> None:
        self.plt.clear_data()
        self.plt.clear_figure()

        if not x_vals or not y_vals:
            self.plt.title("No plottable data")
            self.refresh()
            return

        if len(x_vals) != len(y_vals):
            self.plt.title("q_list and quantile_values have different lengths")
            self.refresh()
            return

        self.plt.title(title)
        self.plt.plot(x_vals, y_vals)

        if fence_low is not None:
            self.plt.plot(x_vals, [fence_low] * len(x_vals))
        if fence_high is not None:
            self.plt.plot(x_vals, [fence_high] * len(x_vals))

        self.refresh()


class QuadsViewer(App):
    CSS = """
    Screen {
        layout: vertical;
    }

    #right {
        height: 1fr;
    }

    #file_summary {
        height: 10;
        border: solid green;
        padding: 1;
    }

    #nan_title {
        height: 1;
        padding: 0 1;
    }

    #nan_table {
        height: 8;
        border: solid white;
    }

    #row_summary {
        height: 13;
        border: solid yellow;
        padding: 1;
    }

    #preview {
        height: 6;
        border: solid cyan;
    }

    #bottom_area {
        height: 1fr;
        layout: horizontal;
    }

    #plot_right {
        width: 1fr;
        border: solid blue;
    }

    #raw_panel {
        width: 1fr;
        border: solid magenta;
    }

    #raw_title {
        height: 1;
        padding: 0 1;
    }

    #raw_table {
        height: 1fr;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "reload_files", "Reload files"),
        ("enter", "activate_current", "Select"),
        ("t", "focus_table", "Preview Table"),
        ("d", "focus_raw_table", "Raw Data"),
        ("s", "save_raw_data", "Save raw data"),
    ]

    REQUIRED_COLUMNS = [
        "id_string",
        "no_of_violations_left",
        "no_of_violations_right",
        "no_of_total_violations",
        "fence_low",
        "fence_high",
        "quantile_values",
        "q_list",
        "is_positive",
        "is_fractional",
        "is_high_priority",
    ]

    RAW_COLUMNS = [
        "time",
        "lat",
        "lon",
        "level",
        "raw data",
        "is_positive",
        "is_fractional",
        "is_high_priority",
    ]
    MAX_RAW_ROWS = 1000

    NAN_COLUMNS = ["Collection", "NaN Count"]
    NAN_CSV_FILENAME = "nan_count_by_collection.csv"

    @dataclass
    class PickleLoaded(Message):
        path: Path
        df: pd.DataFrame

    @dataclass
    class RawDataLoaded(Message):
        row_idx: int
        rows: list[tuple[str, ...]]

    @dataclass
    class RawDataFailed(Message):
        row_idx: int
        message: str

    def __init__(self, root: Path, model: str, date_str: str) -> None:
        super().__init__()
        self.root = root.expanduser().resolve()
        print(self.root)
        self.model = model
        self.date_str = date_str

        self.files: list[Path] = []
        self.df: pd.DataFrame | None = None
        self.view_df: pd.DataFrame | None = None
        self.current_path: Path | None = None
        self.parquet_path: Path | None = None
        self.current_id_string: str | None = None

    def compose(self) -> ComposeResult:
        yield Header()

        with Vertical(id="right"):
            yield Static("Loading .pkl file", id="file_summary")
            yield Static(
                "NaN counts by collection (nonzero, sorted descending)",
                id="nan_title",
            )
            yield DataTable(id="nan_table")
            yield DataTable(id="preview")
            yield Static("Select a row to summarize and plot", id="row_summary")
            with Horizontal(id="bottom_area"):
                yield QuantilePlot(id="plot_right")
                with Vertical(id="raw_panel"):
                    yield Static(
                        'Raw data outside the fence (press "s" to save as CSV)',
                        id="raw_title",
                    )
                    yield DataTable(id="raw_table")

        yield Footer()

    def on_mount(self) -> None:
        preview = self.query_one("#preview", DataTable)
        preview.cursor_type = "row"
        preview.zebra_stripes = True

        raw_table = self.query_one("#raw_table", DataTable)
        raw_table.cursor_type = "row"
        raw_table.zebra_stripes = True
        raw_table.add_columns(*self.RAW_COLUMNS)

        nan_table = self.query_one("#nan_table", DataTable)
        nan_table.cursor_type = "row"
        nan_table.zebra_stripes = True

        self.reload_file_list()
        self.load_nan_counts()
        self.query_one("#plot_right", QuantilePlot).show_message("Select a file")

    def action_reload_files(self) -> None:
        self.reload_file_list()
        self.load_nan_counts()

    def action_focus_table(self) -> None:
        self.query_one("#preview", DataTable).focus()

    def action_focus_raw_table(self) -> None:
        self.query_one("#raw_table", DataTable).focus()

    def action_activate_current(self) -> None:
        focused = self.focused

        if focused is None:
            return

        if focused.id == "preview":
            table = self.query_one("#preview", DataTable)
            row_idx = table.cursor_row
            if row_idx is not None and row_idx >= 0:
                self.update_selected_row(row_idx)

    def action_save_raw_data(self) -> None:
        if self.current_path is None:
            self.notify("No pickle file is currently loaded.", severity="warning")
            return

        if self.parquet_path is None:
            self.notify("No parquet file found next to the selected pickle file.", severity="warning")
            return

        if self.current_id_string is None:
            self.notify("No raw data row is currently selected.", severity="warning")
            return

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.current_id_string)
        out_path = Path.cwd() / f"raw_data_{safe_name}.csv"

        self.save_raw_data(self.current_id_string, self.parquet_path, out_path)

    def reload_file_list(self) -> None:
        expected_date = self.date_str.replace("-", "_")
        self.files = sorted(
            path
            for path in self.root.rglob("*.pkl")
            if (
                (match := FILE_RE.match(path.name))
                and match.group("model").lower() == self.model.lower()
                and match.group("date") == expected_date
            )
        )

        self.query_one("#file_summary", Static).update(
            f"Root: {self.root}\n"
            f"Found {len(self.files)} matching pickle files\n"
            f"Model: {self.model.upper()}\n"
            f"Date: {self.date_str}\n\n"
            f"Press 't' for preview table, 'd' for raw data table, 's' to save raw data."
        )

        if self.files:
            self.load_pickle(self.files[0])
        else:
            self.query_one("#row_summary", Static).update("No pickle files found")
            self.query_one("#plot_right", QuantilePlot).show_message("No pickle files found")
            self.clear_raw_table()

    def load_nan_counts(self) -> None:
        table = self.query_one("#nan_table", DataTable)
        table.clear(columns=True)
        table.add_columns(*self.NAN_COLUMNS)

        csv_path = self.root / self.NAN_CSV_FILENAME

        if not csv_path.exists():
            self.notify(
                f"{csv_path} not found; NaN counts panel left empty.",
                severity="warning",
            )
            return

        try:
            nan_df = pd.read_csv(csv_path)
        except Exception as e:
            self.notify(f"Failed to read {csv_path}: {e}", severity="error")
            return

        if "collection" not in nan_df.columns or "no_of_nans" not in nan_df.columns:
            self.notify(
                f"{csv_path} must have 'collection' and 'no_of_nans' columns.",
                severity="error",
            )
            return

        nonzero = nan_df[nan_df["no_of_nans"] != 0].sort_values(
            by="no_of_nans", ascending=False
        )

        for _, row in nonzero.iterrows():
            table.add_row(str(row["collection"]), f"{int(row['no_of_nans']):,}")

    @work(thread=True, exclusive=True)
    def load_pickle(self, path: Path) -> None:
        try:
            obj = pd.read_pickle(path)

            if not isinstance(obj, pd.DataFrame):
                self.notify(f"{path.name} is not a pandas DataFrame", severity="error")
                return

            missing = [c for c in self.REQUIRED_COLUMNS if c not in obj.columns]
            if missing:
                self.notify(
                    f"{path.name} missing required columns: {missing}",
                    severity="error",
                )
                return

            self.post_message(self.PickleLoaded(path=path, df=obj))

        except Exception as e:
            self.notify(f"Failed to load {path.name}: {e}", severity="error")

    @on(PickleLoaded)
    def handle_pickle_loaded(self, event: PickleLoaded) -> None:
        self.current_path = event.path
        self.df = event.df
        self.view_df = None
        self.current_id_string = None

        expected_date = self.date_str.replace("-", "_")
        parquet_files = sorted(
            path
            for path in self.current_path.parent.glob("*.parquet")
            if (
                (match := PARQUET_RE.match(path.name))
                and match.group("model").lower() == self.model.lower()
                and match.group("date") == expected_date
            )
        )
        self.parquet_path = parquet_files[0] if parquet_files else None

        self.update_file_summary()
        self.update_preview()
        self.update_row_summary(None)
        self.clear_raw_table()

        table = self.query_one("#preview", DataTable)
        if self.view_df is not None and len(self.view_df) > 0:
            table.focus()
            table.move_cursor(row=0)
            self.update_selected_row(0)
        else:
            self.query_one("#plot_right", QuantilePlot).show_message("No data to plot")

    def update_selected_row(self, row_idx: int) -> None:
        self.update_row_summary(row_idx)
        self.plot_row(row_idx)
        self.load_raw_data_for_row(row_idx)

    def update_file_summary(self) -> None:
        assert self.df is not None
        assert self.current_path is not None

        match = FILE_RE.match(self.current_path.name)
        if match:
            model = match.group("model").upper()
            date = match.group("date").replace("_", "-")
        else:
            model = self.model.upper()
            date = self.date_str

        total_left = int(self.df["no_of_violations_left"].sum())
        total_right = int(self.df["no_of_violations_right"].sum())
        total_total = int(self.df["no_of_total_violations"].sum())
        nonzero_rows = int((self.df["no_of_total_violations"] != 0).sum())
        raw_data_file = self.parquet_path.name if self.parquet_path is not None else "Not found"

        text = (
            f"Model: {model}, Date: {date}\n"
            f"Loaded summary file: {self.current_path.name}\n"
            f"Loaded raw data file: {raw_data_file}\n"
            f"Violations below lower T-digest fence: {total_left:,}\n"
            f"Violations above upper T-digest fence: {total_right:,}\n"
            f"Total violations: {total_total:,} across {nonzero_rows:,} data slices (identified by id_strings)\n"
            f"\nPress 't' for preview table, 'd' for raw data table, 's' to save raw data."
        )
        self.query_one("#file_summary", Static).update(text)

    def update_preview(self) -> None:
        assert self.df is not None

        table = self.query_one("#preview", DataTable)
        table.clear(columns=True)

        table.add_columns(
            "id_string (Collection|Variable|Level|Latitude Stratum)",
            "no_of_total_violations",
            "no_of_violations_left",
            "no_of_violations_right",
        )

        self.view_df = self.df[
            self.df["no_of_total_violations"] != 0
        ].sort_values(
            by=["is_high_priority", "no_of_total_violations"],
            ascending=[False, False],
        ).copy()

        for _, row in self.view_df.iterrows():
            table.add_row(
                shorten(row["id_string"], 90),
                str(row["no_of_total_violations"]),
                str(row["no_of_violations_left"]),
                str(row["no_of_violations_right"]),
            )

    @on(DataTable.RowHighlighted, "#preview")
    def handle_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        row_idx = event.cursor_row
        if row_idx is not None and row_idx >= 0:
            self.update_selected_row(row_idx)

    def update_row_summary(self, row_idx: int | None) -> None:
        if (
            self.view_df is None
            or row_idx is None
            or row_idx < 0
            or row_idx >= len(self.view_df)
        ):
            self.query_one("#row_summary", Static).update(
                "Select a row to summarize and plot"
            )
            return

        row = self.view_df.iloc[row_idx]

        text = (
            f"id_string: {row['id_string']} (Collection|Variable|Level|Latitude Stratum)\n"
            f"Total violations: {row['no_of_total_violations']}\n"
            f"Violations below lower T-digest fence: {row['no_of_violations_left']}\n"
            f"Violations above upper T-digest fence: {row['no_of_violations_right']}\n"
            f"fence_low: {row['fence_low']}\n"
            f"fence_high: {row['fence_high']}\n"
            f"is_positive: {row['is_positive']}\n"
            f"is_fractional: {row['is_fractional']}\n"
            f"is_high_priority: {row['is_high_priority']}\n"
        )
        self.query_one("#row_summary", Static).update(text)

    def plot_row(self, row_idx: int) -> None:
        if self.view_df is None or row_idx < 0 or row_idx >= len(self.view_df):
            return

        row = self.view_df.iloc[row_idx]

        x_vals = to_1d_float_list(row["q_list"])
        y_vals = to_1d_float_list(row["quantile_values"])

        fence_low = None
        fence_high = None

        try:
            fence_low = float(row["fence_low"])
        except Exception:
            pass

        try:
            fence_high = float(row["fence_high"])
        except Exception:
            pass

        self.query_one("#plot_right", QuantilePlot).show_row_plot(
            x_vals=x_vals,
            y_vals=y_vals,
            title="Historical reference quantiles and T-digest fence",
            fence_low=fence_low,
            fence_high=fence_high,
        )

    def clear_raw_table(self) -> None:
        table = self.query_one("#raw_table", DataTable)
        table.clear(columns=True)
        table.add_columns(*self.RAW_COLUMNS)
        table.refresh()

    def populate_raw_table(
        self,
        rows: list[tuple[str, ...]],
    ) -> None:
        table = self.query_one("#raw_table", DataTable)
        table.clear(columns=True)
        table.add_columns(*self.RAW_COLUMNS)

        for row in rows:
            table.add_row(*row)

        if rows:
            table.move_cursor(row=0)

        table.refresh()

    def load_raw_data_for_row(self, row_idx: int) -> None:
        if self.view_df is None or row_idx < 0 or row_idx >= len(self.view_df):
            return

        if self.parquet_path is None:
            self.current_id_string = None
            self.clear_raw_table()
            self.notify("No parquet file found next to the selected pickle file.", severity="warning")
            return

        id_string = str(self.view_df.iloc[row_idx]["id_string"])
        self.current_id_string = id_string

        self.clear_raw_table()
        self.load_raw_data(row_idx, id_string, self.parquet_path)

    @work(thread=True, exclusive=True)
    def load_raw_data(self, row_idx: int, id_string: str, parquet_path: Path) -> None:
        try:
            query = f"""
                SELECT
                    COALESCE(CAST(time AS VARCHAR), 'NA') AS time,
                    lat,
                    lon,
                    lev,
                    value AS raw_data,
                    is_positive,
                    is_fractional,
                    is_high_priority
                FROM read_parquet(?)
                WHERE id_string = ?
                LIMIT {self.MAX_RAW_ROWS}
            """

            with duckdb.connect(database=":memory:") as con:
                raw_df = con.execute(query, [str(parquet_path), id_string]).df()

            rows: list[tuple[str, ...]] = []
            for _, row in raw_df.iterrows():
                rows.append(
                    (
                        str(row["time"]),
                        str(row["lat"]),
                        str(row["lon"]),
                        str(row["lev"]),
                        str(row["raw_data"]),
                        str(row["is_positive"]),
                        str(row["is_fractional"]),
                        str(row["is_high_priority"]),
                    )
                )

            self.post_message(self.RawDataLoaded(row_idx=row_idx, rows=rows))

        except Exception as e:
            self.post_message(
                self.RawDataFailed(
                    row_idx=row_idx,
                    message=f"Failed to load raw parquet data: {e}",
                )
            )

    @work(thread=True, exclusive=True)
    def save_raw_data(self, id_string: str, parquet_path: Path, out_path: Path) -> None:
        try:
            query = """
                SELECT
                    COALESCE(CAST(time AS VARCHAR), 'NA') AS time,
                    lat,
                    lon,
                    lev,
                    value AS raw_data,
                    is_positive,
                    is_fractional,
                    is_high_priority
                FROM read_parquet(?)
                WHERE id_string = ?
            """

            with duckdb.connect(database=":memory:") as con:
                raw_df = con.execute(query, [str(parquet_path), id_string]).df()

            raw_df.to_csv(out_path, index=False)

            self.notify(
                f"Saved {len(raw_df):,} raw data rows to {out_path}",
                severity="information",
            )

        except Exception as e:
            self.notify(f"Failed to save raw data: {e}", severity="error")

    @on(RawDataLoaded)
    def handle_raw_data_loaded(self, event: RawDataLoaded) -> None:
        preview = self.query_one("#preview", DataTable)
        current_row = preview.cursor_row

        if current_row is None or current_row != event.row_idx:
            return

        self.populate_raw_table(event.rows)

    @on(RawDataFailed)
    def handle_raw_data_failed(self, event: RawDataFailed) -> None:
        preview = self.query_one("#preview", DataTable)
        current_row = preview.cursor_row

        if current_row is None or current_row != event.row_idx:
            return

        self.clear_raw_table()
        self.notify(event.message, severity="error")


if __name__ == "__main__":
    base_root = Path(
        sys.argv[1]
        if len(sys.argv) > 1
        else "/home/sadhika8/JupyterLinks/nobackup/quads_results"
    ).expanduser()

    model = input("Enter model name (e.g. geosfp, geosit, geoscf, merra2): ").strip()
    date_str = input("Enter date (YYYY-MM-DD, e.g. 2024-02-01): ").strip()

    try:
        parsed_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        print("Date must be in YYYY-MM-DD format.")
        raise SystemExit(1)

    year = f"{parsed_date.year:04d}"
    month = f"{parsed_date.month:02d}"
    day = f"{parsed_date.day:02d}"

    root = base_root / model.upper() / year / month / day
    #print("root:", root)
    print("Opening Quads TUI")
    app = QuadsViewer(
        root=root,
        model=model,
        date_str=date_str,
    )
    app.run()
