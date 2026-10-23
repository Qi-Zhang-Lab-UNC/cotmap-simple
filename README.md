# cotmap-simple.py

Turns ShapeMapper `*_profile.txt` files from a co-transcriptional probing experiment into normalized reactivity matrices, summary plots, and per-transcript outputs. It handles background subtraction, normalization, replicate comparison, and depth reporting.

## What it does

1. Reads all ShapeMapper profile files in a folder (one file per transcript).
2. Builds matrices of nucleotide identity, sequence, read depth, and reactivity (transcripts × nucleotide positions).
3. Optionally subtracts a background dataset.
4. Normalizes reactivity per nucleotide type (A, C, G, U).
5. Optionally combines a second replicate into an average and standard deviation.
6. Writes matrices, plots, and `.dms` files, and prints a read-depth report.

## Requirements

- Python 3
- `numpy`, `pandas`, `scipy`, `matplotlib`, `seaborn`

## Required inputs

| Argument | Description |
|---|---|
| `-i`, `--input` | Folder containing ShapeMapper `*profile.txt` files |
| `-ml`, `--MinMax_length MIN MAX` | Smallest and largest transcript numbers to keep, after the offset is applied |

### Input file expectations

- Filenames must end in `profile.txt`.
- The transcript number is read from the second-to-last underscore-separated field of the filename, and it must be an integer. For example, `..._037_profile.txt` is read as transcript 37.
- The script reads columns 0, 1, 4 and 23 (0-indexed) of each file. In standard ShapeMapper2 output these are nucleotide number, sequence, modified effective depth, and reactivity profile. Check this against your ShapeMapper version.
- The name of the folder holding the profile files does not matter. Pass whichever folder contains them.

## Optional inputs

| Argument | Default | Description |
|---|---|---|
| `-r`, `--replicates` | none | Folder of profile files for a second replicate. Enables the replicate comparison plot, averaging, and error bars. |
| `-b`, `--background` | none | Folder of profile files to subtract from the main dataset and the replicate (negative values are set to 0). |
| `-nf`, `--nt_factors` | calculated | Single-line CSV of normalization factors in the order `A,C,G,U`. If omitted, factors are calculated from the data. |
| `-vm`, `--vmax` | 1.5 | Upper limit of the reactivity color scale. |
| `-offt`, `--offset_transcript` | 0 | Offset subtracted from transcript and nucleotide numbering. Use a negative value (e.g. `-19`) to shift numbering upward. |
| `-d`, `--delta` | none | Folder for a comparison ("delta") dataset. See limitations. |
| `-dr`, `--delta_replicates` | none | Replicate folder for the delta dataset. |
| `-hi`, `--hist` | none | Accepted but not currently used. |
| `-s`, `--show` | True | Accepted but not currently used. Plots are always saved, never displayed, so you can leave it out. |

## Examples

Run from the folder that contains your sample directories, using relative paths.

**Minimal (single dataset):**

```bash
python cotmap-simple.py -i sample1_rep1/profiles -ml 10 305
```

**Two replicates:**

```bash
python cotmap-simple.py \
    -i sample1_rep1/profiles \
    -r sample1_rep2/profiles \
    -ml 10 305 \
    -vm 1
```

**Replicates, offset, and a delta (comparison) dataset:**

```bash
python cotmap-simple.py \
    -i  sample1_rep1/profiles \
    -r  sample1_rep2/profiles \
    -d  sample2_rep1/profiles \
    -dr sample2_rep2/profiles \
    -offt -19 \
    -ml 30 246 \
    -vm 1
```

**Background subtraction and fixed normalization factors:**

```bash
python cotmap-simple.py \
    -i sample1_rep1/profiles \
    -r sample1_rep2/profiles \
    -b background/profiles \
    -nf norm_factors.csv \
    -ml 10 305
```

### Using wildcards

Shell wildcards work, but each one must match **exactly one** folder. If a pattern matches several (for example, profiles collected on different dates), the extra paths will cause an argument error.

```bash
python cotmap-simple.py -i sample1_rep1/profiles_2024* -ml 30 246
```

Check what a pattern matches before running:

```bash
ls -d sample1_rep1/profiles_2024*
```

## Running on SLURM (array job)

This example processes several sample pairs in parallel, one per array task. Replicate 1 folders end in `1` and replicate 2 folders end in `2`, so the replicate name is derived by swapping the last character.

```bash
#!/bin/bash
#SBATCH --job-name=cotmap
#SBATCH --output=slurm-%A_%a.out
#SBATCH --error=slurm-%A_%a.err
#SBATCH --array=0-2%3            # one task per sample (0-indexed), max 3 at once
#SBATCH -p general
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=2g
#SBATCH -t 02:00:00

module purge
module load anaconda
conda activate your_env          # environment with the Python requirements

# Replicate-1 folders and their matching delta (comparison) folders.
# Both lists must be the same length and in the same order.
samples=( sampleA_rep1 sampleB_rep1 sampleC_rep1 )
deltas=(  controlA_rep1 controlB_rep1 controlC_rep1 )

SAMPLE=${samples[$SLURM_ARRAY_TASK_ID]}
DELTA=${deltas[$SLURM_ARRAY_TASK_ID]}

# Replicate 2 = same name with the last character changed to "2"
SAMPLE2=${SAMPLE:0:-1}2
DELTA2=${DELTA:0:-1}2

echo "Processing: $SAMPLE (replicate: $SAMPLE2, delta: $DELTA / $DELTA2)"

python cotmap-simple.py \
    -i  ${SAMPLE}/profiles_2024* \
    -r  ${SAMPLE2}/profiles_2024* \
    -d  ${DELTA}/profiles_2024* \
    -dr ${DELTA2}/profiles_2024* \
    -offt -19 \
    -ml 30 246 \
    -vm 1
```

Submit with `sbatch run_cotmap.sh`. Set `--array` to `0-(N-1)`, where N is the number of samples.

## Outputs

Everything is written to `<first folder of --input>/output_YYYY-MM-DD_IndivNorm_Final/`. In the SLURM example above, each sample gets its own output folder inside its own directory.

| Output | Contents |
|---|---|
| `matrix.pdf/.png` | Read-depth bars plus normalized reactivity heatmap |
| `matrix_avg.pdf/.png` | Same plot using the replicate average |
| `<rep>_vs_<input>ReplicateComparison.pdf/.png` | Depth, per-transcript correlation, and scatter plot (with `-r`) |
| `TranscriptSlices/Transcript_<n>.pdf` | Reactivity bar plot for each transcript |
| `DMS_files/<n>.dms` | One reactivity file per transcript |
| `NormalizedReactivityMatrix_cleaned.csv` | Final averaged matrix |
| `ProcessedData/` | CSVs of the raw, subtracted (with `-b`), normalized, average, and standard deviation matrices, plus depth, sequence, and nucleotide matrices |

The read-depth report (transcripts meeting depth thresholds of 1,000 and 5,000 reads) is printed to the terminal, or to the SLURM `.out` file.

## How normalization works

For each nucleotide type, the factor is the mean reactivity of values between the 90th and 95th percentile. Reactivities of that nucleotide are divided by this factor. The standard deviation between replicates is used for the error bars.

## Known limitations

- **Run with relative paths.** The output folder is named from the first component of the `--input` path. An absolute path (starting with `/`) will try to write to the filesystem root.
- **`--replicates` is currently required in practice.** The depth report for "Replicate 2" runs unconditionally and fails if no replicate was given. Wrapping that call in `if args.replicates is not None:` fixes this.
- **Delta analysis is not active.** The delta data is loaded and normalized, but the functions that use it (`analyze_nucleotide_differences`, `plot_cumulative_significant_fraction`) are commented out in `main()`, so `-d` and `-dr` currently produce no extra output. If you re-enable them, add `from ast import literal_eval` to the imports.
- **`NucleotideSlices/` is empty** because the per-nucleotide plotting loop is commented out.
- Depth thresholds are set inside `report_depth_statistics` and cannot be changed from the command line.
