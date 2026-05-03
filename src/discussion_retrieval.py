import requests
import zipfile
from pathlib import Path
from tqdm import tqdm

NWS_ARCHIVE = "https://mesonet.agron.iastate.edu/cgi-bin/afos/retrieve.py"
DISCUSSION_DIR = "discussions"
OUTPUT_DIR = f"{DISCUSSION_DIR}/trimmed"

start_date = "2026-04-01T00:00Z"
end_date = "2026-04-18T23:59Z"

def read_zip(start_date=start_date, end_date=end_date):
    request_params = {
        "pil": "AFDMTR",
        "fmt": "zip",
        "sdate": start_date, 
        "edate": end_date,
        "limit": 9999
    }

    with requests.get(NWS_ARCHIVE, params=request_params, stream=True) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))

        out_filename = Path(f"{DISCUSSION_DIR}/afdmtr-{start_date}-{end_date}.zip")
        out_filename.parent.mkdir(exist_ok=True, parents=True)
        with open(out_filename, "wb") as f, \
            tqdm(total=total, unit="B", unit_scale=True) as bar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                bar.update(len(chunk))

def process_zip(start_date=start_date, end_date=end_date):
    filename = f"{DISCUSSION_DIR}/afdmtr-{start_date}-{end_date}.zip"
    if not Path(filename).is_file():
        print(f"File {filename} doesn't exist. Fetching file...")
        read_zip(start_date, end_date)

    with zipfile.ZipFile(filename, "r") as zf:
        for fname in zf.namelist():
            date = fname.split("_")[1][:-4]
            out_fname = Path(f"{OUTPUT_DIR}/discussion_{date}")

            if not out_fname.is_file():
                with zf.open(fname) as f:
                    contents = f.read().decode("utf-8")
                    if not "...New SHORT TERM, LONG TERM" in contents:
                        # Not an updated discussion
                        continue

                    sections = contents.split("&&")
                    short_term = sections[1].strip()
                    long_term = sections[2].strip()
                    discussion = "\n\n".join([short_term, long_term])

                out_fname.parent.mkdir(exist_ok=True, parents=True)
                with open(out_fname, "w") as out_file:
                    out_file.write(discussion)

if __name__ == "__main__":
    process_zip(start_date, end_date)
