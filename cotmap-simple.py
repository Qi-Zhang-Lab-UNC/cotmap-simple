#!/usr/bin/env python3

import csv
import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import datetime
from scipy.stats import pearsonr
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    sns.set()
    args = parse_arguments()
    output_dir, transcripts_dir, nucleotides_dir, proc_dir = make_directories(args.input)

    nucleotide_df, sequence_df, depth_df, reactivity_df = parse_data_into_dataframes(
        args.input, args.offset_transcript, args.MinMax_length[0], args.MinMax_length[1]
    )
    reactivity_df.to_csv(proc_dir + "RawReactivityMatrix.csv")


    # --- Background subtraction ---
    if args.background is not None:
        _, _, _, background_reactivity_df = parse_data_into_dataframes(
            args.background, args.offset_transcript, args.MinMax_length[0], args.MinMax_length[1]
        )
        reactivity_df = (reactivity_df - background_reactivity_df).clip(lower=0)
        print("Background subtracted using:", args.background)
        reactivity_df.to_csv(proc_dir + "SubtractedRawReactivityMatrix.csv")

    # --- Replicate handling ---
    replicate_reactivity_df = None
    replicate_depth_df = None
    if args.replicates is not None:
        _, _, replicate_depth_df, replicate_reactivity_df = parse_data_into_dataframes(
            args.replicates, args.offset_transcript, args.MinMax_length[0], args.MinMax_length[1]
        )
        if args.background is not None:
            replicate_reactivity_df = replicate_reactivity_df.subtract(background_reactivity_df).clip(lower=0)
            print("Replicate was background subtracted using:", args.background)

        replicate_plots(reactivity_df, replicate_reactivity_df, depth_df, replicate_depth_df,
                        args.MinMax_length[0], args.MinMax_length[1])
        label = args.replicates.split("/")[0] + "_vs_" + args.input.split("/")[0]
        plt.savefig(output_dir + label + "ReplicateComparison.pdf")
        plt.savefig(output_dir + label + "ReplicateComparison.png", dpi=300)
        # plt.show()
        plt.close()
    report_depth_statistics(depth_df, "Replicate 1")
    report_depth_statistics(replicate_depth_df, "Replicate 2")

    # --- Normalization ---
    if args.nt_factors is not None:
        try:
            nt_factors = np.loadtxt(args.nt_factors[0], delimiter=',')
            print(nt_factors)
            print("Loaded Norm Factors:", nt_factors)
            norm_factors = {"A": nt_factors[0], "C": nt_factors[1], "G": nt_factors[2], "U": nt_factors[3]}
            reactivity_df = apply_norm_factors(reactivity_df, sequence_df, norm_factors)
        except Exception as e:
            print(f"Error: {type(e).__name__}: {e}")
    else:
        norm_factors = calculate_norm_factors(reactivity_df, sequence_df)
        reactivity_df = apply_norm_factors(reactivity_df, sequence_df, norm_factors)
        print("Applied Norm Factors:", norm_factors)

    # --- Matrix Plotting ---
    plot_HistoMatrix(depth_df, reactivity_df, sequence_df, args.vmax)
    plt.savefig(output_dir + "matrix.pdf")
    plt.savefig(output_dir + "matrix.png", dpi=300)
    plt.close()

    # --- Combine replicates or create zero std ---
    if replicate_reactivity_df is not None:
        if args.nt_factors is not None:
            try:
                replicate_nt_factors = np.loadtxt(args.nt_factors[0], delimiter=',')
                replicate_norm_factors = {"A": replicate_nt_factors[0], "C": replicate_nt_factors[1], "G": replicate_nt_factors[2], "U": replicate_nt_factors[3]}
                replicate_reactivity_df = apply_norm_factors(replicate_reactivity_df, sequence_df, replicate_norm_factors)
                print("Loaded Replicate Norm Factors:", norm_factors)
                reactivity_df_avg, _, reactivity_df_std, depth_df_std = combine_replicates(
                reactivity_df, replicate_reactivity_df, depth_df, replicate_depth_df)
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")
        else:
            replicate_norm_factors = calculate_norm_factors(replicate_reactivity_df, sequence_df)
            replicate_reactivity_df = apply_norm_factors(replicate_reactivity_df, sequence_df, replicate_norm_factors)
            print("Replicate Norm Factors:", replicate_norm_factors)
            
            reactivity_df_avg, _, reactivity_df_std, depth_df_std = combine_replicates(
                reactivity_df, replicate_reactivity_df, depth_df, replicate_depth_df)
    else:
        reactivity_df_avg = reactivity_df
        reactivity_df_std = pd.DataFrame(0, index=reactivity_df.index, columns=reactivity_df.columns)
        print("No Replicates Found, Creating Zero Standard Deviations")
    
    plt.close('all')

    # --- Delta Set Processing ---
    if args.delta is not None:
        _, _, delta_depth_df, delta_reactivity_df = parse_data_into_dataframes(
            args.delta, args.offset_transcript, args.MinMax_length[0], args.MinMax_length[1]
        )

        # --- Background subtraction ---
        if args.background is not None:    
            delta_reactivity_df = (delta_reactivity_df - background_reactivity_df).clip(lower=0)
            print("Background subtracted using:", args.background)

        # --- Replicate handling ---
        delta_replicate_reactivity_df = None
        delta_replicate_depth_df = None
        if args.delta_replicates is not None:
            _, _, delta_replicate_depth_df, delta_replicate_reactivity_df = parse_data_into_dataframes(
                args.delta_replicates, args.offset_transcript, args.MinMax_length[0], args.MinMax_length[1]
            )
            if args.background is not None:
                delta_replicate_reactivity_df = delta_replicate_reactivity_df.subtract(background_reactivity_df).clip(lower=0)
                print("Delta Replicate was background subtracted using:", args.background)

        # --- Normalization ---
        if args.nt_factors is not None:
            try:
                nt_factors = np.loadtxt(args.nt_factors[0], delimiter=',')
                print(nt_factors)
                print("Loaded Norm Factors towards Delta set:", nt_factors)
                delta_norm_factors = {"A": nt_factors[0], "C": nt_factors[1], "G": nt_factors[2], "U": nt_factors[3]}
                delta_reactivity_df = apply_norm_factors(delta_reactivity_df, sequence_df, delta_norm_factors)
            except Exception as e:
                print(f"Error: {type(e).__name__}: {e}")
        else:
            delta_norm_factors = calculate_norm_factors(delta_reactivity_df, sequence_df)
            delta_reactivity_df = apply_norm_factors(delta_reactivity_df, sequence_df, delta_norm_factors)
            print("Applied Norm Factors towards Delta set:", delta_norm_factors)

        # --- Combine replicates or create zero std ---
        if delta_replicate_reactivity_df is not None:
            if args.nt_factors is not None:
                try:
                    replicate_nt_factors = np.loadtxt(args.nt_factors[0], delimiter=',')
                    delta_replicate_norm_factors = {"A": replicate_nt_factors[0], "C": replicate_nt_factors[1], "G": replicate_nt_factors[2], "U": replicate_nt_factors[3]}
                    delta_replicate_reactivity_df = apply_norm_factors(delta_replicate_reactivity_df, sequence_df, delta_replicate_norm_factors)
                    print("Loaded Replicate Norm Factors:", delta_norm_factors)
                    delta_reactivity_df_avg, _, delta_reactivity_df_std, _ = combine_replicates(
                    delta_reactivity_df, delta_replicate_reactivity_df, delta_depth_df, delta_replicate_depth_df)
                except Exception as e:
                    print(f"Error: {type(e).__name__}: {e}")
            else:
                delta_replicate_norm_factors = calculate_norm_factors(delta_replicate_reactivity_df, sequence_df)
                delta_replicate_reactivity_df = apply_norm_factors(delta_replicate_reactivity_df, sequence_df, delta_replicate_norm_factors)
                print("Replicate Norm Factors:", delta_replicate_norm_factors)
                
                delta_reactivity_df_avg, _, delta_reactivity_df_std, _ = combine_replicates(
                    delta_reactivity_df, delta_replicate_reactivity_df, delta_depth_df, delta_replicate_depth_df)
        else:
            delta_reactivity_df_avg = delta_reactivity_df
            delta_reactivity_df_std = pd.DataFrame(0, index=delta_reactivity_df.index, columns=delta_reactivity_df.columns)
            print("No Delta Replicates Found, Creating Zero Standard Deviations")

    # --- Save processed data ---
    generate_dms_files(reactivity_df_avg, output_dir, args.offset_transcript)
    nucleotide_df.to_csv(proc_dir + "NucleotideMatrix.csv")
    sequence_df.to_csv(proc_dir + "SequenceMatrix.csv")
    depth_df.to_csv(proc_dir + "DepthMatrix.csv")
    reactivity_df.to_csv(proc_dir + "NormalizedReactivityMatrix.csv")
    reactivity_df_std.to_csv(proc_dir+"NormalizedReactivityMatrixSTD.csv")
    reactivity_df_avg.to_csv(proc_dir+"NormalizedReactivityMatrixAVG.csv")




    plot_HistoMatrix(depth_df, reactivity_df_avg, sequence_df, args.vmax)
    plt.savefig(output_dir + "matrix_avg.pdf")
    plt.savefig(output_dir + "matrix_avg.png", dpi=300)
    # plt.show()
    plt.close()


    # calculate_differential_band(reactivity_df, depth_df, replicate_reactivity_df, replicate_depth_df,sequence_df, args.MinMax_length[0], args.MinMax_length[1])
    # plt.show()

    # quit()

    # --- Per-nucleotide slice plots (parallelized) ---
    vmax = float(args.vmax)
    offset = int(args.offset_transcript)
    min_len, max_len = args.MinMax_length
    # maximums_df, distribution_df, fig, ax = (
    # plot_maximum_position_distribution(
    #         reactivity_df=reactivity_df_avg,
    #         sequence_df=sequence_df,
    #         n_relative_positions=16,
    #         output_path=(
    #             output_dir + "RelativeMaximumPositionDistribution.pdf"
    #         ),
    #     )
    # )
    # # quit()
    # if args.delta is not None:
    #     nucleotide_difference_df = analyze_nucleotide_differences(
    #         reactivity_df_avg=reactivity_df_avg,
    #         reactivity_df_std=reactivity_df_std,
    #         delta_reactivity_df_avg=delta_reactivity_df_avg,
    #         delta_reactivity_df_std=delta_reactivity_df_std,
    #         sequence_df=sequence_df,
    #         nucleotides_dir=nucleotides_dir,
    #         min_len=min_len,
    #         max_len=max_len,
    #         offset=offset,
    #         vmax=vmax,
    #     )

    #     fig, ax, cumulative_df, sigmoid_parameters = plot_cumulative_significant_fraction(
    #         nucleotide_difference_df,
    #         output_path=(output_dir + "CumulativeSignificantFraction_Total.pdf"),
    #         min_len=min_len,
    #         max_len=max_len)

    #     HMP_residues = [12, 13, 14, 34, 35, 36, 37, 38, 39]
    #     fig, ax, cumulative_df, sigmoid_parameters = plot_cumulative_significant_fraction(
    #         nucleotide_difference_df,
    #         residue_positions=HMP_residues,
    #         output_path=(output_dir + "CumulativeSignificantFraction_HMP.pdf"),
    #         min_len=min_len,
    #         max_len=max_len)

    #     PPate_residues = [51, 52, 53, 54, 77, 78, 79, 80, 81]
    #     fig, ax, cumulative_df, sigmoid_parameters = plot_cumulative_significant_fraction(
    #         nucleotide_difference_df,
    #         residue_positions=PPate_residues,
    #         output_path=(output_dir + "CumulativeSignificantFraction_PPate.pdf"),
    #         min_len=min_len,
    #         max_len=max_len)

    #     tert_residues = [55, 56, 57, 58, 69, 70, 73, 74, 75]
    #     fig, ax, cumulative_df, sigmoid_parameters = plot_cumulative_significant_fraction(
    #         nucleotide_difference_df,
    #         residue_positions=tert_residues,
    #         output_path=(output_dir + "CumulativeSignificantFraction_tert.pdf"),
    #         min_len=min_len,
    #         max_len=max_len)

    # quit()
    def save_nucleotide_plot(i, fig, ax):
        col = reactivity_df_avg.columns[i + offset]
        nucleotide_type = sequence_df.iloc[:, i + offset].value_counts().idxmax()
        fig, ax = plt.subplots(1, 1, figsize=(12, 3))
        Nucleotide_Slice_withError_window(
            reactivity_df_avg[col].tolist(),
            reactivity_df_avg.index.tolist(),
            vmax, nucleotide_type, i, ax,
            reactivity_df_std[col].tolist()
        )
        ax.set_xlim(min_len, max_len)
        ax.set_ylim(-0.05, 2.5)
        plt.savefig(nucleotides_dir + f"Nucleotide_{nucleotide_type}{i}.pdf")
        plt.cla()


    def save_transcript_plot(i, fig, ax):
        # fig, ax = plt.subplots(1, 1, figsize=(10, 3))
        Transcript_Slice_avg(
            reactivity_df_avg.loc[i].tolist(),
            reactivity_df_std.loc[i].tolist(),
            sequence_df.loc[i].tolist(),
            vmax, i, offset, ax
        )
        ax.set_xlim(0, max_len + offset)
        plt.savefig(transcripts_dir + f"Transcript_{i}.pdf")
        plt.cla()


    plt.close('all')

    # fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    # for i in range(len(reactivity_df_avg.columns)):
    #     save_nucleotide_plot(i, fig, ax)
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    for i in reactivity_df_avg.index:
        save_transcript_plot(i, fig, ax)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str)
    parser.add_argument("--hist", "-hi", required=False, type=str)
    parser.add_argument("--vmax", "-vm", required=False, type=str, default=1.5)
    parser.add_argument("--nt_factors", "-nf", nargs=1, required=False, type=str, help="Single line CSV file containing norm factors for [A, C, G, U] in that order.")
    parser.add_argument("--offset_transcript", "-offt", required=False, type=int, default=0)
    parser.add_argument("--MinMax_length", "-ml", required=True, nargs=2, type=int)
    parser.add_argument("--delta", "-d", required=False, type=str)
    parser.add_argument("--show", "-s", required=False, type=bool, default=True)
    parser.add_argument("--replicates", "-r", required=False, type=str)
    parser.add_argument("--delta_replicates", "-dr", required=False, type=str)
    parser.add_argument("--background", "-b", required=False, type=str)
    return parser.parse_args()


def make_directories(input_name):
    base = input_name.split("/")[0]
    output_dir = f"{base}/output_{datetime.date.today():%Y-%m-%d}_IndivNorm_Final/"
    subdirs = {
        'transcripts': output_dir + "TranscriptSlices/",
        'nucleotides': output_dir + "NucleotideSlices/",
        'proc':        output_dir + "ProcessedData/",
    }
    for d in [output_dir] + list(subdirs.values()):
        os.makedirs(d, exist_ok=True)
    return output_dir, subdirs['transcripts'], subdirs['nucleotides'], subdirs['proc']


def apply_norm_factors(reactivity_df, sequence_df, norm_factors):
    """Vectorized normalization — avoids repeated boolean indexing in a loop."""
    result = reactivity_df.copy()
    for n, factor in norm_factors.items():
        mask = sequence_df == n
        result[mask] = result[mask] / factor
    return result


def calculate_norm_factors(reactivity_df, sequence_df):
    norm_factor_dict = {}
    for n in ('A', 'C', 'G', 'U'):
        mask = sequence_df == n
        # Select cells where the nucleotide matches, drop all-NaN columns
        n_reactivity = reactivity_df[mask].dropna(axis=1, how='all')
        profile = n_reactivity.to_numpy().ravel()
        profile = profile[~np.isnan(profile)]
        bnds = np.percentile(profile, [90., 95.])
        pmask = (profile >= bnds[0]) & (profile < bnds[1])
        n1 = np.mean(profile[pmask])
        # n2 = np.percentile(profile[profile > 0.001], 75)
        # norm_factor_dict[n] = max(n1, n2)
        norm_factor_dict[n] = max(n1, 0)
    return norm_factor_dict


def generate_dms_files(reactivity_df, output_path, offset):
    reactivity_df.to_csv(output_path + "NormalizedReactivityMatrix_cleaned.csv")
    dms_dir = output_path + "DMS_files/"
    os.makedirs(dms_dir, exist_ok=True)

    pad = np.full(27, -999, dtype=float)

    def write_dms(args):
        index, row = args
        row_vals = row.iloc[1:index + offset + 1].values
        full = np.concatenate([pad, row_vals, pad])
        lines = "\n".join(
            f"{i+1}  {str(v).replace('-999.0', '-999')}" for i, v in enumerate(full)
        )
        with open(dms_dir + f"{index}.dms", "w") as f:
            f.write(lines + "\n")

    with ThreadPoolExecutor() as executor:
        executor.map(write_dms, reactivity_df.iterrows())


def parse_data_into_dataframes(data_folder, offset, minimum_length, maximum_length):
    """Parse all profile.txt files into DataFrames, using vectorized ops throughout."""
    records = {}
    for file_name in os.listdir(data_folder):
        if not file_name.endswith('profile.txt'):
            continue
        transcript_number = int(file_name.split('_')[-2])
        file_path = os.path.join(data_folder, file_name)
        data = pd.read_csv(file_path, sep='\t', header=0, usecols=[0, 1, 4, 23])
        records[transcript_number] = {
            'nucleotide': data.iloc[:, 0].tolist(),
            'sequence':   data.iloc[:, 1].tolist(),
            'depth':      data.iloc[:, 2].tolist(),
            'reactivity': data.iloc[:, 3].tolist(),
        }

    def build_df(key):
        df = pd.DataFrame.from_dict({k: v[key] for k, v in records.items()}, orient='index')
        # Fill missing transcript indices
        for i in range(-offset, maximum_length - offset):
            if i not in df.index:
                df.loc[i] = np.nan
        df.sort_index(inplace=True)
        df.drop(df.columns[:26], axis=1, inplace=True)
        df.columns = df.columns - 26 - offset
        df.drop(df.columns[-26:], axis=1, inplace=True)
        df.index = df.index - offset - 1
        return df

    nucleotide_df = build_df('nucleotide')
    sequence_df   = build_df('sequence')
    depth_df      = build_df('depth')
    reactivity_df = build_df('reactivity')

    reactivity_df.clip(lower=0, inplace=True)

    # Slice to desired length range
    idx = slice(int(minimum_length), int(maximum_length))
    return nucleotide_df.loc[idx], sequence_df.loc[idx], depth_df.loc[idx], reactivity_df.loc[idx]


def combine_replicates(reactivity_df_1, reactivity_df_2, depth_df_1, depth_df_2):
    reactivity_avg = (reactivity_df_1 + reactivity_df_2) / 2
    depth_avg      = (depth_df_1 + depth_df_2) / 2
    reactivity_std = np.sqrt(((reactivity_df_1 - reactivity_avg)**2 + (reactivity_df_2 - reactivity_avg)**2) / 2)
    depth_std      = np.sqrt(((depth_df_1 - depth_avg)**2      + (depth_df_2 - depth_avg)**2) / 2)
    return reactivity_avg, depth_avg, reactivity_std, depth_std


def replicate_plots(reactivity_df, reference_reactivity_df, depth_df, reference_depth_df, tmin, tmax):
    common_keys    = sorted(set(reactivity_df.index) & set(reference_reactivity_df.index))
    common_columns = sorted(set(reactivity_df.columns) & set(reference_reactivity_df.columns))

    r1 = reactivity_df.loc[common_keys, common_columns]
    r2 = reference_reactivity_df.loc[common_keys, common_columns]

    arr1 = r1.to_numpy().ravel()
    arr2 = r2.to_numpy().ravel()

    # Drop NaN pairs in one pass
    valid = ~(np.isnan(arr1) | np.isnan(arr2)) & (arr1 > 0) & (arr2 > 0)
    clean1, clean2 = arr1[valid], arr2[valid]

    r, _ = pearsonr(clean1, clean2)

    corr_series = r1.corrwith(r2, axis=1)   # per-row Pearson, vectorized
    corr_df = corr_series.rename('Correlation').to_frame()
    corr_df['Color'] = np.where(
        corr_df['Correlation'] > 0.80, 'mediumseagreen',
        np.where(corr_df['Correlation'] > 0.50, 'goldenrod', 'tomato')
    )

    reference_depth_dict = make_abs_depth_dic(reference_depth_df)
    depth_dict           = make_abs_depth_dic(depth_df)

    sns.set()
    fig, axs = plt.subplots(1, 4, figsize=(13, 3))
    colors = ["black", "dimgrey"]
    for ax, dd, title, color in zip(axs[:2], [reference_depth_dict, depth_dict], ["Replicate 1", "Replicate 2"], colors):
        ax.barh(list(dd.keys()), list(dd.values()), color=color, alpha=0.98, height=1, linewidth=0.0)
        ax.set_xlabel("Normalized Read Depth")
        ax.set_ylabel("Transcript Number")
        ax.xaxis.set_label_position('bottom')
        ax.xaxis.tick_bottom()
        ax.set_title(title)

    max_x = max(axs[0].get_xlim()[1], axs[1].get_xlim()[1])
    for ax in axs[:2]:
        ax.set_xlim(-0.001 * max_x, max_x)
    axs[0].invert_xaxis()
    axs[0].invert_yaxis()
    axs[0].yaxis.tick_left()
    axs[0].spines['right'].set_visible(False)
    axs[1].spines['left'].set_visible(False)
    axs[1].yaxis.tick_right()
    axs[1].yaxis.label.set_fontsize(0)
    axs[1].yaxis.set_tick_params(labelsize=0)
    # axs[1].sharey(axs[0])
    axs[1].invert_yaxis()

    axs[2].barh(corr_df.index, corr_df['Correlation'], color=corr_df['Color'], height=1, linewidth=0.1)
    axs[2].set_ylim(tmin, tmax)
    axs[2].set_xlim(-0.1, 1.1)
    axs[2].invert_yaxis()
    axs[2].set_xlabel("Corr. Coeff. per Transcript")
    axs[2].xaxis.set_label_position('top')
    axs[2].yaxis.label.set_fontsize(0)
    axs[2].yaxis.set_tick_params(labelsize=0)
    axs[2].sharey(axs[0])
    axs[2].xaxis.tick_bottom()
    axs[2].axvline(x=0, color='black', linewidth=0.5)

    bins = [100, 100]
    hh, locx, locy = np.histogram2d(np.log10(clean1), np.log10(clean2), bins=bins)
    # Vectorized density lookup
    ix = np.searchsorted(locx[1:], np.log10(arr1))
    iy = np.searchsorted(locy[1:], np.log10(arr2))
    ix = np.clip(ix, 0, hh.shape[0] - 1)
    iy = np.clip(iy, 0, hh.shape[1] - 1)
    z  = hh[ix, iy]
    idx_sort = z.argsort()
    sc = axs[3].scatter(arr1[idx_sort], arr2[idx_sort], c=z[idx_sort], s=10, cmap='viridis')
    # fig.colorbar(sc, ax=axs[3], shrink=0.1)

    lim = [min(axs[3].get_ylim()[0], axs[3].get_xlim()[0]),
           max(axs[3].get_ylim()[1], axs[3].get_xlim()[1])]
    # axs[3].set_xlim(*lim); axs[3].set_ylim(*lim)
    axs[3].plot(lim, lim, 'k--', linewidth=0.5)
    axs[3].set_title(f'Total r = {r:.2f}, n = {len(clean1)}')
    # put a textbox in the upper left hand corner of the r value and the number of points
    # axs[3].text(0.05, 0.95, f'Total r = {r:.2f}, n = {len(clean1)}', transform=axs[3].transAxes)
    axs[3].set_xlabel("Replicate 1 Mut. Rate")
    axs[3].set_ylabel("Replicate 2 Mut. Rate")
    axs[3].set_aspect('equal', 'box')
    axs[3].locator_params(axis='x', nbins=3)
    axs[3].locator_params(axis='y', nbins=3)
    axs[3].axhline(0, color='black', linewidth=0.5)
    axs[3].axvline(0, color='black', linewidth=0.5)
    axs[3].yaxis.tick_right()
    axs[3].xaxis.tick_bottom()
    axs[3].yaxis.set_label_position("right")
    axs[3].set_xscale('log')
    axs[3].set_yscale('log')

    plt.subplots_adjust(wspace=0.001)
    return fig, axs


def make_depth_dic(depth_df):
    max_vals = depth_df.max(axis=1).fillna(0)
    total = max_vals.sum()
    return (max_vals / total).to_dict()
def make_abs_depth_dic(depth_df):
    max_vals = depth_df.max(axis=1).fillna(0)
    return (max_vals).to_dict()

def color_list(reactivity_list, vmax):
    cm = plt.colormaps["plasma"]
    arr = np.array(reactivity_list, dtype=float)
    norm = np.clip(arr / vmax, 0, 1)
    return [cm(v) for v in norm]


def plot_HistoMatrix(depth_df, reactivity_df, sequence_df, v_maximum):
    fig, (ax1, ax2) = plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 10]}, figsize=(10, 5))

    depth_max = depth_df.max(axis=1)
    ax1.barh(depth_df.index - depth_df.index[0], depth_max, color="black", linewidth=0, height=1)
    ax1b = ax1.twiny()
    norm_depth = depth_max / depth_max.sum()
    ax1b.barh(norm_depth.index - norm_depth.index[0], norm_depth, color="gray", linewidth=0, height=1, alpha=0.75)

    ax1.set_xlabel("Absolute Read Depth")
    ax1.set_ylabel("Transcript Number")
    ax1.xaxis.set_label_position('top')
    ax1.xaxis.tick_top()
    ax1.invert_xaxis()
    ax1.invert_yaxis()
    ax1.yaxis.tick_left()
    ax1b.xaxis.set_label_position('bottom')
    ax1b.xaxis.tick_bottom()
    ax1b.set_xlabel("Relative Read Depth")
    ax1b.invert_xaxis()

    ax2.imshow(reactivity_df, cmap="plasma", vmin=0, vmax=v_maximum, aspect='equal', origin="upper")
    # ax2b = ax2.twiny()
    # ax2b.imshow(reactivity_df, cmap="plasma", vmin=0, vmax=v_maximum, aspect='equal', origin="upper")
    ax2.set_xlabel("Nucleotide")
    ax1.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.xaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.yaxis.set_major_locator(plt.MultipleLocator(20))
    ax2.yaxis.set_tick_params(rotation=0)
    ax2.xaxis.tick_bottom()
    ax2.xaxis.set_tick_params(rotation=90)
    ax2.tick_params(axis='both', which='both', length=5)


    ax1b.sharey(ax1)
    # ax2b.sharey(ax2)
    
    # ax2b.xaxis.set_label_position('top')
    # ax2b.xaxis.tick_top()
    # ax2b.xaxis.set_tick_params(rotation=90)
    # ax2b.tick_params(axis='both', which='both', length=5)


    ax1.spines['left'].set_visible(True)
    ax1.spines['top'].set_visible(True)
    ax2.spines['left'].set_visible(False)
    ax2.spines['right'].set_visible(True)
    ax1b.spines['bottom'].set_visible(True)
    # ax2b.spines['bottom'].set_visible(True)
    # ax2b.spines['left'].set_visible(False)
    # ax2b.spines['right'].set_visible(False)

    ax2.tick_params(labelleft=True)
    ax2.yaxis.tick_left()

    # for ax in [ax1, ax2, ax1b, ax2b]:
    for ax in [ax1, ax2, ax1b]:
        ax.grid(False)
        ax.patch.set_visible(False)
        sns.despine(ax=ax, left=True, bottom=True, right=True, top=True)

    fig.patch.set_visible(False)
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.001)
    return fig, ax1, ax2, ax1b


# def Transcript_Slice_avg(reactivity_df_transcript, reactivity_df_transcript_std, sequence_df_transcript, vmax, transcript_number, offset, ax):
#     cm = plt.colormaps["inferno"]
#     labels = [
#         (str(n) + str(i - offset)) if n is not None else ""
#         for i, n in enumerate(sequence_df_transcript)
#     ]

#     colors = color_list(reactivity_df_transcript, vmax)
#     sns.set()
#     sns.despine()
#     sns.set_style("ticks")
#     bars = ax.bar(labels, reactivity_df_transcript, linewidth=0.1, width=1)
#     ax.errorbar(labels, reactivity_df_transcript, yerr=reactivity_df_transcript_std,
#                 fmt='.', elinewidth=1, capsize=0, ms=0, ecolor="black", barsabove=False)
#     ax.text(0.01, 0.95, f'Transcript {int(transcript_number)}',
#             transform=ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold')
#     ax.xaxis.set_visible(False)
#     ax.xaxis.label.set_visible(True)
#     for bar, c in zip(bars, colors):
#         bar.set_color(c)
#         bar.set_edgecolor('black')
#     ax.set_xlim(-0.5, len(sequence_df_transcript) - 0.5)
#     ax.set_ylim(vmax * (-0.05), 2)
#     ax.set_xlabel('Nucleotide')
#     ax.set_ylabel('Normalized Reactivity')
#     return ax
def Transcript_Slice_avg(reactivity_df_transcript, reactivity_df_transcript_std, sequence_df_transcript, vmax, transcript_number, offset, ax):
    
    # ── Dimension check ───────────────────────────────────────────────────────
    lengths = {
        "reactivity": len(reactivity_df_transcript),
        "std":        len(reactivity_df_transcript_std),
        "sequence":   len(sequence_df_transcript),
    }
    min_len = min(lengths.values())
    if len(set(lengths.values())) > 1:
        print(
            f"WARNING: Transcript {transcript_number} — mismatched lengths "
            f"{lengths}. Clipping all to {min_len}."
        )
        reactivity_df_transcript       = reactivity_df_transcript[:min_len]
        reactivity_df_transcript_std   = reactivity_df_transcript_std[:min_len]
        sequence_df_transcript         = sequence_df_transcript[:min_len]
    # ─────────────────────────────────────────────────────────────────────────

    cm = plt.colormaps["inferno"]
    labels = [
        (str(n) + str(i - offset)) if n is not None else ""
        for i, n in enumerate(sequence_df_transcript)
    ]
    colors = color_list(reactivity_df_transcript, vmax)
    sns.set()
    sns.despine()
    sns.set_style("ticks")
    bars = ax.bar(labels, reactivity_df_transcript, linewidth=0.1, width=1)
    ax.errorbar(labels, reactivity_df_transcript, yerr=reactivity_df_transcript_std,
                fmt='.', elinewidth=1, capsize=0, ms=0, ecolor="black", barsabove=False)
    ax.text(0.01, 0.95, f'Transcript {int(transcript_number)}',
            transform=ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold')
    ax.xaxis.set_visible(True)
    ax.xaxis.label.set_visible(True)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=3, ha='center')
    for bar, c in zip(bars, colors):
        bar.set_color(c)
        bar.set_edgecolor('black')
    ax.set_xlim(-0.5, len(sequence_df_transcript) - 0.5)
    ax.set_ylim(vmax * (-0.05), 2)
    ax.set_xlabel('Nucleotide')
    ax.set_ylabel('Normalized Reactivity')
    return ax

def Nucleotide_Slice_withError_window(reactivity_df_nucleotide, reactivity_df_index, vmax, nucleotide_type, nucleotide_number, ax, reactivity_df_std):
    # Clip negatives in-place efficiently
    reactivity_arr = np.maximum(reactivity_df_nucleotide, 0)

    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_color('black')
    ax.spines['left'].set_color('black')
    ax.xaxis.label.set_color('black')
    ax.yaxis.label.set_color('black')
    ax.grid(linewidth=0.15)
    ax.spines['bottom'].set_linewidth(0.44)
    ax.spines['left'].set_linewidth(0.44)
    sns.set_style("white")
    sns.set_style("ticks")
    sns.despine()

    ax.errorbar(reactivity_df_index, reactivity_arr, yerr=reactivity_df_std,
                fmt='.', elinewidth=.44, capthick=0.25, capsize=4, ms=0,
                ecolor="seagreen", barsabove=False, mfc="black", alpha=0.95, zorder=10)
    ax.scatter(reactivity_df_index, reactivity_arr, c=reactivity_arr, vmin=0, vmax=vmax,
               cmap="plasma", linewidth=.44, edgecolors="seagreen", s=85/3, alpha=0.95, zorder=11)
    ax.scatter(reactivity_df_index, reactivity_arr, c=reactivity_arr, vmin=0, vmax=vmax,
               cmap="plasma", linewidth=0, s=85/3, alpha=0.95, zorder=12)
    ax.plot(reactivity_df_index, reactivity_arr, color='seagreen', linewidth=0.11, zorder=13)

    ax.set_xlabel('Transcript Ending in Nucleotide (TEN)')
    ax.set_ylabel('Norm. Reactivity')
    ax.text(0.03, 0.95, f"{nucleotide_type}{nucleotide_number}",
            transform=ax.transAxes, fontsize=12, verticalalignment='top', fontweight='bold')
    return ax


def analyze_nucleotide_differences(
    reactivity_df_avg,
    reactivity_df_std,
    delta_reactivity_df_avg,
    delta_reactivity_df_std,
    sequence_df,
    nucleotides_dir,
    min_len,
    max_len,
    offset,
    vmax=2.5,
):
    def filter_contiguous_positions(positions, minimum_run_length=4):
        if not positions:
            return []

        positions = sorted(set(int(position) for position in positions))

        retained_positions = []
        current_run = [positions[0]]

        for position in positions[1:]:
            if position == current_run[-1] + 1:
                current_run.append(position)
            else:
                if len(current_run) >= minimum_run_length:
                    retained_positions.extend(current_run)

                current_run = [position]

        # Check the final run
        if len(current_run) >= minimum_run_length:
            retained_positions.extend(current_run)

        return retained_positions

    nucleotides_dir = Path(nucleotides_dir)
    nucleotides_dir.mkdir(parents=True, exist_ok=True)

    results = []

    # for i, col in enumerate(reactivity_df_avg.columns):
    for i, col in enumerate(range(min_len, max_len)):

        # Data for one nucleotide position across transcript positions
        reactivity = np.maximum(
            reactivity_df_avg[col].to_numpy(dtype=float),
            0,
        )

        delta_reactivity = np.maximum(
            delta_reactivity_df_avg[col].to_numpy(dtype=float),
            0,
        )

        reactivity_std = reactivity_df_std[col].to_numpy(dtype=float)
        delta_std = delta_reactivity_df_std[col].to_numpy(dtype=float)

        transcript_positions = np.asarray(
            reactivity_df_avg.index
        )

        # Determine nucleotide identity
        nucleotide_type = (
            sequence_df[col]
            .dropna()
            .mode()
            .iloc[0]
        )

        # Difference and combined standard deviation
        difference = reactivity - delta_reactivity

        combined_std = reactivity_std + delta_std

        # Avoid division by zero without adding extensive validation logic.
        combined_std[combined_std == 0] = np.nan

        # Simplified two-sided test statistic
        test_value = (1- 1.96* combined_std / np.abs(difference))
        # test_value = (1- 1.65* combined_std / np.abs(difference))

        significant = (test_value > 0)

        significant_positions = transcript_positions[significant].tolist()


        significant_positions = filter_contiguous_positions(
            significant_positions,
            minimum_run_length=4,
        )

        # Reconstruct the Boolean mask after the contiguous-run filter so that the
        # plotting highlights exactly the positions retained in the output list.
        significant = np.isin(
            transcript_positions,
            significant_positions,
        )

        offset_position = col+offset-1
        results.append(
            {
                "nucleotide_position": offset_position,
                "nucleotide_type": nucleotide_type,
                "significant_transcript_positions": (
                    significant_positions
                ),
                "n_significant": len(significant_positions),
            }
        )

        # -------------------------------------------------------------
        # Plot using the Nucleotide_Slice_withError_window style
        # -------------------------------------------------------------

        sns.set_style("white")
        sns.set_style("ticks")

        fig, ax = plt.subplots(
            1,
            1,
            figsize=(10, 2),
        )

        ax.spines["bottom"].set_visible(True)
        ax.spines["left"].set_visible(True)
        ax.spines["bottom"].set_color("black")
        ax.spines["left"].set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.grid(linewidth=0.15)
        ax.spines["bottom"].set_linewidth(0.44)
        ax.spines["left"].set_linewidth(0.44)

        sns.despine(ax=ax)

        # Main reactivity dataset
        ax.errorbar(
            transcript_positions,
            reactivity,
            yerr=reactivity_std,
            fmt=".",
            elinewidth=0.44*2,
            capthick=0.25*2,
            capsize=4,
            ms=0,
            ecolor="seagreen",
            alpha=0.95,
            zorder=10,
        )

        ax.scatter(
            transcript_positions,
            reactivity,
            c=reactivity,
            vmin=0,
            vmax=vmax,
            cmap="plasma",
            linewidth=0.44*2,
            edgecolors="seagreen",
            s=85 / 5,
            alpha=0.95,
            zorder=11,
        )

        ax.scatter(
            transcript_positions,
            reactivity,
            c="seagreen",
            linewidth=0,
            s=85 / 5,
            alpha=0.95,
            zorder=12,
        )


        # Delta dataset
        ax.errorbar(
            transcript_positions,
            delta_reactivity,
            yerr=delta_std,
            fmt=".",
            elinewidth=0.44*2,
            capthick=0.25*2,
            capsize=4,
            ms=0,
            ecolor="tomato",
            alpha=0.90,
            zorder=14,
        )

        ax.scatter(
            transcript_positions,
            delta_reactivity,
            c=delta_reactivity,
            vmin=0,
            vmax=vmax,
            cmap="plasma",
            linewidth=0.44*2,
            edgecolors="tomato",
            s=85 / 5,
            alpha=0.90,
            zorder=15,
        )

        ax.scatter(
            transcript_positions,
            delta_reactivity,
            c="tomato",
            linewidth=0,
            s=85 / 5,
            alpha=0.90,
            zorder=16,
        )


        # Highlight significant transcript positions
        if np.any(significant):

            significant_x = transcript_positions[significant]

            for position in significant_x:
                ax.axvspan(
                    position - 0.5,
                    position + 0.5,
                    facecolor="tan",
                    edgecolor="none",
                    alpha=0.5,
                    zorder=0,
                )

        ax.set_xlim(min_len, max_len)
        ax.set_ylim(-0.05, vmax*3)

        ax.set_xlabel(
            "TEN"
        )
        ax.set_ylabel(r'$\rho$')
        offset_position = col+offset-1
        ax.text(
            0.03,
            0.95,
            f"{nucleotide_type}{offset_position}",
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            fontweight="bold",
        )

        ax.legend(
            loc="upper left",
            frameon=False,
        )

        fig.tight_layout()


        fig.savefig(
            nucleotides_dir
            / f"Nucleotide_{nucleotide_type}{offset_position}_DeltaHighlights_2SD.pdf",
            bbox_inches="tight",
        )
        
        plt.close(fig)

    return pd.DataFrame(results)


def plot_cumulative_significant_fraction(
    nucleotide_difference_df,
    residue_positions=None,
    output_path=None,
    ax=None,
    ax_derivative=None,
    show_derivative=True,
    min_len=0,
    max_len=230,
):
 
    required_columns = {
        "nucleotide_position",
        "significant_transcript_positions",
    }
 
    missing_columns = required_columns.difference(
        nucleotide_difference_df.columns
    )
 
    if missing_columns:
        raise KeyError(
            "nucleotide_difference_df is missing required columns: "
            f"{sorted(missing_columns)}"
        )
 
    def parse_positions(value):
        """
        Convert a list-like cell into a list of integer positions.
        """
        if value is None:
            return []
 
        if isinstance(value, float) and np.isnan(value):
            return []
 
        if isinstance(value, str):
            value = value.strip()
 
            if not value or value == "[]":
                return []
 
            try:
                value = literal_eval(value)
            except (ValueError, SyntaxError):
                value = value.strip("[]")
                value = value.split(",")
 
        if isinstance(value, (int, np.integer)):
            return [int(value)]
 
        return [
            int(position)
            for position in value
            if pd.notna(position)
        ]
 
    # Determine the first significant transcript coordinate for every
    # nucleotide. Nucleotides with no significant positions remain NaN.
    # The denominator always includes every residue in the input DataFrame.
    total_nucleotides = len(nucleotide_difference_df)

    if total_nucleotides == 0:
        raise ValueError(
            "nucleotide_difference_df contains no nucleotide rows."
        )

    # Select which residues are allowed to contribute to the cumulative
    # numerator. If None, use all residues.
    if residue_positions is None:
        residues_for_count = nucleotide_difference_df
    else:
        residue_positions = {
            int(position)
            for position in residue_positions
        }

        residue_numbers = pd.to_numeric(
            nucleotide_difference_df["nucleotide_position"],
            errors="coerce",
        )

        residues_for_count = nucleotide_difference_df.loc[
            residue_numbers.isin(residue_positions)
        ]

    # Determine the first significant transcript coordinate only for the
    # selected residues.
    first_significant_positions = []

    for _, row in residues_for_count.iterrows():
        significant_positions = parse_positions(
            row["significant_transcript_positions"]
        )

        if significant_positions:
            first_position = min(significant_positions)
        else:
            first_position = np.nan

        first_significant_positions.append(first_position)

    first_positions = np.asarray(
        first_significant_positions,
        dtype=float,
    )
    valid_first_positions = first_positions[
        np.isfinite(first_positions)
    ]

    # Tabulate every transcriptional coordinate through the requested endpoint,
    # even after the cumulative fraction stops changing.
    coordinates = np.arange(
        int(min_len),
        int(max_len) + 1,
    )

    cumulative_counts = np.array(
        [
            np.sum(valid_first_positions <= coordinate)
            if len(valid_first_positions) > 0
            else 0
            for coordinate in coordinates
        ],
        dtype=float,
    )

    # The denominator remains all residues in the original input DataFrame.
    cumulative_fraction = (
        cumulative_counts / total_nucleotides
    )
    # valid_first_positions = first_positions[
    #     np.isfinite(first_positions)
    # ]

    # if len(valid_first_positions) == 0:
    #     raise ValueError(
    #         "None of the selected residues has a statistically "
    #         "significant position."
    #     )
 
    # # Use each unique first-significant coordinate as a cumulative x value.
    # # Include every transcript coordinate, not only coordinates where the
    # # cumulative count changes.
    # coordinates = np.arange(
    #     int(valid_first_positions.min()),
    #     int(valid_first_positions.max()) + 1,
    # )

    # cumulative_counts = np.array(
    #     [
    #         np.sum(valid_first_positions <= coordinate)
    #         for coordinate in coordinates
    #     ],
    #     dtype=float,
    # )

    # cumulative_fraction = (
    #     cumulative_counts / total_nucleotides
    # )
    first_positions = np.asarray(
        first_significant_positions,
        dtype=float,
    )

    # Analyze only the selected residues in the numerator, while retaining
    # every residue in the denominator.
    valid_first_positions = first_positions[
        np.isfinite(first_positions)
    ]

    # Include every transcriptional coordinate from min_len through max_len.
    # The cumulative fraction remains constant when no new residue becomes
    # significant.
    coordinates = np.arange(
        int(min_len),
        int(max_len) + 1,
    )

    cumulative_counts = np.array(
        [
            np.sum(valid_first_positions <= coordinate)
            if len(valid_first_positions) > 0
            else 0
            for coordinate in coordinates
        ],
        dtype=float,
    )

    cumulative_fraction = (
        cumulative_counts / total_nucleotides
    )

    cumulative_df = pd.DataFrame(
        {
            "transcript_coordinate": coordinates,
            "nucleotides_with_significance": (
                cumulative_counts.astype(int)
            ),
            "total_nucleotides": total_nucleotides,
            "cumulative_fraction": cumulative_fraction,
        }
    )
    # cumulative_df = pd.DataFrame(
    #     {
    #         "transcript_coordinate": coordinates,
    #         "nucleotides_with_significance": cumulative_counts.astype(int),
    #         "total_nucleotides": total_nucleotides,
    #         "cumulative_fraction": cumulative_fraction,
    #     }
    # )
 
    # --- figure / axis setup -----------------------------------------
    if ax is None:
        if show_derivative:
            fig, (ax, ax_derivative) = plt.subplots(
                2, 1,
                figsize=(8, 6.5),
                sharex=True,
                gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
            )
        else:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax_derivative = None
    else:
        fig = ax.figure
        if show_derivative and ax_derivative is None:
            raise ValueError(
                "show_derivative=True requires ax_derivative when ax is "
                "passed in explicitly; leave ax=None to have both axes "
                "created automatically, or pass ax_derivative yourself."
            )
 
    sns.set_style("white")
    sns.set_style("ticks")
 
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_color("black")
    ax.spines["left"].set_color("black")
    ax.spines["bottom"].set_linewidth(0.44)
    ax.spines["left"].set_linewidth(0.44)
    ax.grid(linewidth=0.15)
 
    # Plot the exact empirical cumulative fraction.
    ax.plot(
        cumulative_df["transcript_coordinate"],
        cumulative_df["cumulative_fraction"],
        color="darkslateblue",
        linewidth=1.5,
        marker="o",
        markersize=3,
    )
 
    ax.scatter(
        cumulative_df["transcript_coordinate"],
        cumulative_df["cumulative_fraction"],
        color="darkslateblue",
        s=28,
        zorder=4,
    )
 
    ax.set_ylabel(
        "Cumulative Fraction of Nucleotides\n"
        "with ≥1 Significant Difference"
    )
 
    ax.set_ylim(0, 1.05)
    ax.set_xlim(min_len, max_len)
    ax.set_title("Cumulative Significant-Difference Distribution")
 
    ax.legend(frameon=False, loc="best")
 
    sns.despine(ax=ax)
 
    # --- derivative subplot -------------------------------------------
    if show_derivative:
        derivative = np.gradient(
            cumulative_df["cumulative_fraction"].to_numpy(),
            cumulative_df["transcript_coordinate"].to_numpy(),
        )
        cumulative_df["derivative"] = derivative
 
        ax_derivative.spines["bottom"].set_visible(True)
        ax_derivative.spines["left"].set_visible(True)
        ax_derivative.spines["bottom"].set_color("black")
        ax_derivative.spines["left"].set_color("black")
        ax_derivative.spines["bottom"].set_linewidth(0.44)
        ax_derivative.spines["left"].set_linewidth(0.44)
        ax_derivative.grid(linewidth=0.15)
 
        ax_derivative.axhline(0, color="black", linewidth=0.5, zorder=1)
 
        ax_derivative.fill_between(
            cumulative_df["transcript_coordinate"],
            derivative,
            0,
            step="post",
            color="darkslateblue",
            alpha=0.25,
            zorder=2,
        )
        ax_derivative.step(
            cumulative_df["transcript_coordinate"],
            derivative,
            where="post",
            color="darkslateblue",
            linewidth=1.5,
            zorder=3,
        )
        ax_derivative.scatter(
            cumulative_df["transcript_coordinate"],
            derivative,
            color="darkslateblue",
            s=18,
            zorder=4,
        )
 
        ax_derivative.set_xlabel("Transcript Coordinate")
        ax_derivative.set_ylabel("d(Cum. Frac.)\n/ dCoord")
        ax_derivative.set_xlim(min_len, max_len)
 
        sns.despine(ax=ax_derivative)
 
        # Shared x-axis: keep tick labels only on the bottom subplot.
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
    else:
        ax.set_xlabel("Transcript Coordinate")
 
    fig.tight_layout()
 
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
 
        fig.savefig(
            output_path,
            bbox_inches="tight",
        )
 
    return fig, ax, ax_derivative, cumulative_df
def plot_maximum_position_distribution(
    reactivity_df,
    sequence_df=None,
    n_relative_positions=20,
    output_path=None,
):
    """
    Determine the relative maximum position for every nucleotide and plot
    the distribution of those maximum positions.

    Relative position is calculated as:

        transcript_position - nucleotide_number

    Parameters
    ----------
    reactivity_df : pandas.DataFrame
        Reactivity DataFrame. Rows are transcript coordinates and columns are
        nucleotide sequence numbers.

    sequence_df : pandas.DataFrame, optional
        Sequence DataFrame used to identify nucleotide type.

    n_relative_positions : int, default=30
        Number of relative positions examined for each nucleotide.

    output_path : str or pathlib.Path, optional
        Output path for the distribution plot.

    Returns
    -------
    maximums_df : pandas.DataFrame
        One row per nucleotide with its relative maximum position.

    distribution_df : pandas.DataFrame
        Count and fraction of nucleotide maxima at each relative position.

    fig : matplotlib.figure.Figure
        Distribution figure.

    ax : matplotlib.axes.Axes
        Distribution axis.
    """

    if n_relative_positions <= 0:
        raise ValueError(
            "n_relative_positions must be greater than zero."
        )

    transcript_positions = np.asarray(
        reactivity_df.index,
        dtype=float,
    )

    maximum_results = []

    for column in reactivity_df.columns:

        try:
            nucleotide_number = int(column)
        except (TypeError, ValueError):
            raise ValueError(
                f"Column {column!r} cannot be interpreted as a nucleotide "
                "sequence number."
            )

        relative_positions = (
            transcript_positions - nucleotide_number
        )

        relative_mask = (
            (relative_positions >= 0)
            & (relative_positions < n_relative_positions)
        )

        if not np.any(relative_mask):
            continue

        x_values = relative_positions[relative_mask]

        y_values = pd.to_numeric(
            reactivity_df[column],
            errors="coerce",
        ).to_numpy(dtype=float)[relative_mask]

        valid_values = np.isfinite(y_values)

        if not np.any(valid_values):
            continue

        x_values = x_values[valid_values]
        y_values = y_values[valid_values]

        # Clip negative values consistently with the previous plotting code.
        y_values = np.maximum(y_values, 0)

        # Select the first maximum if multiple positions have the same value.
        maximum_index = np.argmax(y_values)

        relative_maximum_position = int(
            x_values[maximum_index]
        )

        maximum_reactivity = float(
            y_values[maximum_index]
        )

        nucleotide_type = None

        if sequence_df is not None:
            if column in sequence_df.columns:
                sequence_values = sequence_df[column].dropna()
            elif str(column) in sequence_df.columns:
                sequence_values = sequence_df[str(column)].dropna()
            else:
                sequence_values = pd.Series(dtype=object)

            if len(sequence_values) > 0:
                nucleotide_type = str(
                    sequence_values.mode().iloc[0]
                ).upper()

        maximum_results.append(
            {
                "nucleotide_position": nucleotide_number,
                "nucleotide_type": nucleotide_type,
                "relative_maximum_position": (
                    relative_maximum_position
                ),
                "transcript_maximum_position": (
                    nucleotide_number
                    + relative_maximum_position
                ),
                "maximum_reactivity": maximum_reactivity,
            }
        )

    maximums_df = pd.DataFrame(
        maximum_results,
        columns=[
            "nucleotide_position",
            "nucleotide_type",
            "relative_maximum_position",
            "transcript_maximum_position",
            "maximum_reactivity",
        ],
    )

    if maximums_df.empty:
        raise ValueError(
            "No valid nucleotide profiles were available for analysis."
        )

    # Count maxima at every possible relative position, including positions
    # with zero observations.
    all_relative_positions = np.arange(
        n_relative_positions,
    )

    counts = (
        maximums_df["relative_maximum_position"]
        .value_counts()
        .reindex(
            all_relative_positions,
            fill_value=0,
        )
        .sort_index()
    )

    distribution_df = pd.DataFrame(
        {
            "relative_position": all_relative_positions,
            "nucleotide_count": counts.to_numpy(dtype=int),
        }
    )

    distribution_df["fraction"] = (
        distribution_df["nucleotide_count"]
        / len(maximums_df)
    )

    # ------------------------------------------------------------------
    # Plot distribution
    # ------------------------------------------------------------------

    sns.set_style("white")
    sns.set_style("ticks")

    fig, ax = plt.subplots(
        figsize=(9, 5),
    )

    ax.bar(
        distribution_df["relative_position"],
        distribution_df["fraction"],
        width=0.85,
        color="steelblue",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.85,
    )

    mean_position = maximums_df[
        "relative_maximum_position"
    ].mean()

    median_position = maximums_df[
        "relative_maximum_position"
    ].median()

    ax.axvline(
        mean_position,
        color="darkorange",
        linestyle="--",
        linewidth=1.5,
        label=f"Mean = {mean_position:.2f}",
    )

    ax.axvline(
        median_position,
        color="darkgreen",
        linestyle=":",
        linewidth=1.5,
        label=f"Median = {median_position:.1f}",
    )

    ax.set_xlim(
        -0.5,
        n_relative_positions - 0.5,
    )

    ax.set_ylim(
        0,
        max(
            distribution_df["fraction"].max() * 1.15,
            0.05,
        ),
    )

    ax.set_xlabel(
        "Relative Position of Maximum Reactivity"
    )

    ax.set_ylabel(
        "Fraction of Nucleotides"
    )

    ax.set_title(
        "Distribution of Relative Maximum Reactivity Positions"
    )

    ax.set_xticks(
        all_relative_positions
    )

    ax.legend(
        frameon=False,
    )

    ax.grid(
        axis="y",
        linewidth=0.15,
    )

    sns.despine(ax=ax)

    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            output_path,
            bbox_inches="tight",
        )

    return maximums_df, distribution_df, fig, ax

def report_depth_statistics(depth_df, label="Dataset", thresholds=(1000, 5_000), min_length=None, max_length=None):
    """Report read-depth coverage statistics across transcripts.

    For each transcript (row), the median read depth across all nucleotide
    positions is computed. The function then reports what percentage of
    transcripts meet or exceed each depth threshold.

    Parameters
    ----------
    depth_df : pd.DataFrame
        Read-depth matrix (transcripts × nucleotide positions).
    label : str
        Name printed in the report header (e.g. sample or replicate name).
    thresholds : tuple of int
        Depth cutoffs to evaluate. Default: (500, 5_000, 50_000, 500_000).

    Returns
    -------
    summary_df : pd.DataFrame
        One row per threshold with columns:
        threshold, n_passing, n_total, pct_passing, median_of_medians.
    """
    # ── Per-transcript median depth ───────────────────────────────────────────
    # Ignore NaN positions (e.g. nucleotides beyond the transcript's length)
    depth_df = depth_df.replace(0, np.nan)
    if min_length is not None and max_length is not None:
        depth_df = depth_df.loc[min_length:max_length]
    elif min_length is not None:
        depth_df = depth_df.loc[min_length:]
    elif max_length is not None:
        depth_df = depth_df.loc[:max_length]
    # transcript_medians = depth_df.median(axis=1, skipna=True)
    transcript_medians = depth_df.max(axis=1, skipna=True)

    n_total            = len(transcript_medians)
    median_of_medians  = transcript_medians.median()
    mean_of_medians    = transcript_medians.mean()
    min_median         = transcript_medians.min()
    max_median         = transcript_medians.max()

    # ── Threshold table ───────────────────────────────────────────────────────
    rows = []
    for t in thresholds:
        n_passing = int((transcript_medians >= t).sum())
        pct       = 100.0 * n_passing / n_total if n_total > 0 else 0.0
        rows.append({
            "threshold":       t,
            "n_passing":       n_passing,
            "n_total":         n_total,
            "pct_passing":     round(pct, 1),
            "median_of_medians": round(median_of_medians, 1),
        })

    summary_df = pd.DataFrame(rows)

    # ── Printed report ────────────────────────────────────────────────────────
    bar = "=" * 52
    print(f"\n{bar}")
    print(f"  Read Depth Report — {label}")
    print(bar)
    print(f"  Transcripts evaluated : {n_total}")
    print(f"  Median of medians     : {median_of_medians:>12,.1f}")
    print(f"  Mean of medians       : {mean_of_medians:>12,.1f}")
    print(f"  Min median depth      : {min_median:>12,.1f}")
    print(f"  Max median depth      : {max_median:>12,.1f}")
    print(f"\n  {'Threshold':>12}  {'Passing':>8}  {'Total':>8}  {'% Passing':>10}")
    print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*10}")
    for row in rows:
        print(
            f"  {row['threshold']:>12,}  "
            f"{row['n_passing']:>8,}  "
            f"{row['n_total']:>8,}  "
            f"{row['pct_passing']:>9.1f}%"
        )
    print(bar)

    return summary_df
if __name__ == "__main__":
    main()


