from math import ceil, floor
from pathlib import Path
from tqdm import tqdm
import json
import os
import subprocess
import re
import argparse
import psutil
import shutil
import platform
from vstools import Keyframes, vs, core, initialize_clip
core.max_cache_size = 1024

IS_WINDOWS = platform.system() == 'Windows'
NULL_DEVICE = 'NUL' if IS_WINDOWS else '/dev/null'

if shutil.which("av1an") is None:
    raise FileNotFoundError("av1an not found, exiting")

ssimu2zig = True
default_skip = 1

parser = argparse.ArgumentParser()
parser.add_argument("-s", "--stage", help = "Select stage: 1 = encode, 2 = metrics and zones | Default: 0 (all)", default=0)
parser.add_argument("-i", "--input", required=True, help = "Video input filepath (original source file)")
parser.add_argument("-t", "--temp", help = "The temporary directory for av1an to store files in | Default: video input filename")
parser.add_argument("-q", "--quality", help = "Base quality (cq) | Default: 20", default=20)
parser.add_argument("-d", "--deviation", help = "Maximum cq change from original | Default: 4", default=4)
parser.add_argument("-p", "--preset", help = "Fast encode preset | Default: 5", default=5)
parser.add_argument("-w", "--workers", help = "Number of av1an workers | Default: amount of physical cores", default=psutil.cpu_count(logical=False))
parser.add_argument("-S", "--skip", help = "SSIMU2 skip value, every nth frame's SSIMU2 is calculated | Default: 1 for turbo-metrics, 3 for vs-zip")
parser.add_argument("-a", "--aggressive", action='store_true', help = "More aggressive boosting | Default: not active")
args = parser.parse_args()
stage = int(args.stage)
src_file = Path(args.input).resolve()
output_dir = src_file.parent
tmp_dir = Path(args.temp).resolve() if args.temp is not None else output_dir / src_file.stem
output_file = output_dir / f"{src_file.stem}_fastpass.mkv"
scenes_file = tmp_dir / "scenes.json"
br = float(args.deviation)
skip = int(args.skip) if args.skip is not None else default_skip
aggressive = args.aggressive

def get_ranges(scenes: str) -> list[int]:
    """
    Reads a scene file and returns a list of frame numbers for each scene change.

    :param scenes: path to scene file
    :type scenes: str

    :return: list of frame numbers
    :rtype: list[int]
    """
    ranges = [0]
    with scenes.open("r") as file:
        content = json.load(file)
        for scene in content['scenes']:
            ranges.append(scene['end_frame'])
    return ranges

def keyframes_helper(file_path, min_kf_dist, max_kf_dist):

    out = Path(os.path.dirname(file_path)) / f"{Path(file_path).resolve().stem}.cfg"
    clip = core.lsmas.LWLibavSource(r"{}".format(file_path))
    len_clip = len(clip)

    if out.exists():
        print("Reusing existing keyframes config.")
        with open(out, "r", encoding="utf-8") as f:
            keyframes_cut = [int(line.strip()) for line in f.readlines()]
        keyframes_cut.sort()
        return keyframes_cut, len_clip

    print("Generating keyframes config file...")

    keyframes = Keyframes.from_clip(clip, 0)
    keyframes.sort()

    #print(len(keyframes))
    for i, frame in enumerate(keyframes):
        if i == 0:
            keyframes_cut = [str(frame)]
        else:
            frame_diff = keyframes[i]-keyframes[i-1]

            if frame_diff>=min_kf_dist:
                keyframes_cut.append(str(frame))

    #print(len(keyframes_cut))

    keyframes_cut = [int(fr) for fr in keyframes_cut]
    keyframes_cut.sort()
    keyframes_str = f"{'\n'.join([str(i) for i in keyframes_cut])}"
    with open(out, "w", encoding="utf-8") as f:
        f.write(keyframes_str)
    print("Done generating keyframes")
    return keyframes_cut, len_clip

def fast_pass(
        input_file: str, output_file: str, tmp_dir: str, preset: int, cq: int, workers: int
):
    """
    Quick fast-pass using Av1an

    :param input_file: path to input file
    :type input_file: str
    :param output_file: path to output file
    :type output_file: str
    :param tmp_dir: path to temporary directory
    :type tmp_dir: str
    :param preset: encoder preset
    :type preset: int
    :param cq: target cq
    :type cq: int
    :param workers: number of workers
    :type workers: int
    """

    scenes, len_clip = keyframes_helper(input_file, 48, 240)

    string = '{"scenes":['
    for scene_index in range(0, len(scenes)):#-1):
        if scene_index == len(scenes) - 1:
            start = scenes[scene_index]
            end = len_clip - 13
        else:
            start = scenes[scene_index]
            end = scenes[scene_index + 1] - 1
        string += '{'
        string += f'"start_frame":{start},"end_frame":{end + 1},"zone_overrides":'
        string += '{'
        string += f'"encoder":"aom","passes":2,"video_params":["--end-usage=q","--threads=1","--cpu-used={preset}","--cq-level={cq}","--tile-columns=1","--tile-rows=1","--tune-content=psy101","--arnr-strength=2","--arnr-maxframes=7","--input-bit-depth=10","--bit-depth=10","--disable-kf",--kf-min-dist=999","--kf-max-dist=999","--transfer-characteristics=bt709","--color-primaries=bt709","--matrix-coefficients=bt709"],"extra_splits_len":0,"min_scene_len":48'
        string += '}},'
    string = string[:-1]
    string += f'],"frames":{len_clip - 12}'+'}'
    with open(f"{tmp_dir}.log", "w") as f:
        f.write(string)
        
    fast_av1an_command = [
        'av1an',
        '-i', input_file,
        '--temp', tmp_dir,
        '--scenes',f"{tmp_dir}.log",
        '-y',
        '--verbose',
        '--keep',
        '--split-method', 'none',
        '-m', 'lsmash',
        '-c', 'mkvmerge',
        '--min-scene-len', '48',
        '--sc-downscale-height', '720',
        '-e', 'aom',
        '--force',
        '-v', '',
        '-w', str(workers),
        '-x','240',
        '--resume',
        '-o', output_file
    ]

    try:
        subprocess.run(fast_av1an_command, text=True, check=True)
    except subprocess.CalledProcessError as e:
       print(f"Av1an encountered an error:\n{e}")
       exit(1)

def calculate_ssimu2(src_file, enc_file, ssimu2_txt_path, ranges, skip):
    is_vpy = os.path.splitext(os.path.basename(src_file))[1] == ".vpy"
    vpy_vars = {}
    if is_vpy:
        exec(open(src_file).read(), globals(), vpy_vars)
    # in order for auto-boost to use a .vpy file as a source, the output clip should be a global variable named clip
    source_clip = core.lsmas.LWLibavSource(source=src_file, cache=0) if not is_vpy else vpy_vars["clip"]
    encoded_clip = core.lsmas.LWLibavSource(source=enc_file, cache=0)

    source_clip = initialize_clip(source_clip, bits=0)[:-12]#[:34068]
    source_clip = core.resize.Bicubic(source_clip, format=vs.RGBS, matrix_in=1)

    encoded_clip = initialize_clip(encoded_clip, bits=0)
    encoded_clip = core.resize.Bicubic(encoded_clip, format=vs.RGBS, matrix_in=1)

    print(f"source: {len(source_clip)} frames")
    print(f"encode: {len(encoded_clip)} frames")
    with ssimu2_txt_path.open("w") as file:
        file.write(f"skip: 1\n")#{skip}\n")
    iter = 0
    with tqdm(total=floor(len(source_clip) / int(skip)), desc=f'Calculating SSIMULACRA 2 scores') as pbar:
        for i in range(len(ranges) - 1):
            cut_source_clip = source_clip[ranges[i]:ranges[i+1]]#.std.SelectEvery(cycle=skip, offsets=1)
            cut_encoded_clip = encoded_clip[ranges[i]:ranges[i+1]]#.std.SelectEvery(cycle=skip, offsets=1)
            result = cut_source_clip.vship.SSIMULACRA2(cut_encoded_clip)
            for index, frame in enumerate(result.frames()):
                iter += 1
                score = frame.props['_SSIMULACRA2']
                with ssimu2_txt_path.open("a") as file:
                    file.write(f"{iter}: {score}\n")
                pbar.update(skip)

def get_ssimu2(ssimu2_txt_path):
    ssimu2_scores: list[int] = []

    with ssimu2_txt_path.open("r") as file:
        skipmatch = re.search(r"skip: ([0-9]+)", file.readline())
        if skipmatch:
            skip = int(skipmatch.group(1))
        else:
            print("Skip value not detected in SSIMU2 file, exiting.")
            exit(-2)
        for line in file:
            match = re.search(r"([0-9]+): ([0-9]+\.[0-9]+)", line)
            if match:
                score = float(match.group(2))
                ssimu2_scores.append(score)
            else:
                print(line)
    return ssimu2_scores, skip

def calculate_std_dev(score_list: list[int]):
    """
    Takes a list of metrics scores and returns the associated arithmetic mean,
    5th percentile and 95th percentile scores.

    :param score_list: list of SSIMU2 scores
    :type score_list: list
    """

    filtered_score_list = [score if score >= 0 else 0.0 for score in score_list]
    sorted_score_list = sorted(filtered_score_list)
    average = sum(filtered_score_list)/len(filtered_score_list)
    percentile_5 = sorted_score_list[len(filtered_score_list)* 16 // 100] #16th for real
    return (average, percentile_5)

def generate_zones(ranges: list, percentile_5_total: list, average: int, cq: int, zones_txt_path: str):
    """
    Appends a scene change to the ``zones_txt_path`` file in Av1an zones format.

    creates ``zones_txt_path`` if it does not exist. If it does exist, the line is
    appended to the end of the file.

    :param ranges: Scene changes list
    :type ranges: list
    :param percentile_5_total: List containing all 5th percentile scores
    :type percentile_5_total: list
    :param average: Full clip average score
    :type average: int
    :param cq: cq setting to use for the zone
    :type cq: int
    :param zones_txt_path: Path to the zones.txt file
    :type zones_txt_path: str
    """
    string = '{"scenes":['
    zones_iter = 0
    for i in range(len(ranges)-1):
        zones_iter += 1
        if aggressive:
            new_cq = cq - ceil((1.0 - (percentile_5_total[i] / average)) * 40)
        else:
            new_cq = cq - ceil((1.0 - (percentile_5_total[i] / average)) * 20)

        if new_cq < cq - br: # set lowest allowed cq
            new_cq = cq - br

        if new_cq > cq + br: # set highest allowed cq
            new_cq = cq + br

        print(f'Enc:  [{ranges[i]}:{ranges[i+1]}]\n'
              f'Chunk 5th percentile: {percentile_5_total[i]}\n'
              f'Adjusted cq: {new_cq}\n')

        # with zones_txt_path.open("w" if zones_iter == 1 else "a") as file:
        #     file.write(f"{ranges[i]} {ranges[i+1]} svt-av1 --cq {new_cq}\n")

        if i == len(ranges)-1:
            start = ranges[i]
            end = ranges[i+1]
        else:
            start = ranges[i]
            end = ranges[i+1]
        string += '{'
        string += f'"start_frame":{start},"end_frame":{end},"zone_overrides":'
        string += '{'
        string += f'"encoder":"aom","passes":2,"video_params":["--end-usage=q","--threads=1","--cpu-used=2","--cq-level={int(new_cq)}","--enable-dnl-denoising=0","--denoise-noise-level=9","--tile-columns=1","--tile-rows=1","--tune-content=psy101","--arnr-strength=2","--arnr-maxframes=7","--input-bit-depth=10","--bit-depth=10","--disable-kf","--kf-min-dist=999","--kf-max-dist=999","--transfer-characteristics=bt709","--color-primaries=bt709","--matrix-coefficients=bt709"],"extra_splits_len":0,"min_scene_len":48'
        string += '}},'
    string += '{'
    string += f'"start_frame":{ranges[-1]},"end_frame":{len_clip},"zone_overrides":'
    string += '{'
    string += f'"encoder":"aom","passes":2,"video_params":["--end-usage=q","--threads=1","--cpu-used=2","--cq-level={int(new_cq)}","--enable-dnl-denoising=0","--denoise-noise-level=9","--tile-columns=1","--tile-rows=1","--tune-content=psy101","--arnr-strength=2","--arnr-maxframes=7","--input-bit-depth=10","--bit-depth=10","--disable-kf","--kf-min-dist=999","--kf-max-dist=999","--transfer-characteristics=bt709","--color-primaries=bt709","--matrix-coefficients=bt709"],"extra_splits_len":0,"min_scene_len":48'
    string += '}},'
    string = string[:-1]
    string += f'],"frames":{len_clip}'+'}'
    with open(f"{tmp_dir}_final.log", "w") as f:
        f.write(string)

def calculate_metrics(src_file, output_file, tmp_dir, ranges, skip):
    ssimu2_txt_path = output_dir / f"{src_file.stem}_ssimu2.log"
    calculate_ssimu2(src_file, output_file, ssimu2_txt_path, ranges, skip)

def calculate_zones(tmp_dir, ranges, cq):
    ssimu2_txt_path = output_dir / f"{src_file.stem}_ssimu2.log"
    (ssimu2_scores, skip) = get_ssimu2(ssimu2_txt_path)
    ssimu2_zones_txt_path = tmp_dir / "ssimu2_zones.txt"
    ssimu2_total_scores: list[int] = []
    ssimu2_percentile_5_total = []
    ssimu2_iter = 0

    for i in range(len(ranges)-1):
        ssimu2_chunk_scores: list[int] = []
        ssimu2_frames = (ranges[i+1] - ranges[i]) // skip
        for frames in range(ssimu2_frames):
            ssimu2_score = ssimu2_scores[ssimu2_iter]
            ssimu2_chunk_scores.append(ssimu2_score)
            ssimu2_total_scores.append(ssimu2_score)
            ssimu2_iter += 1
        (ssimu2_average, ssimu2_percentile_5) = calculate_std_dev(ssimu2_chunk_scores)
        ssimu2_percentile_5_total.append(ssimu2_percentile_5)
        #print(f'5th Percentile:  {ssimu2_percentile_5}')
    (ssimu2_average, ssimu2_percentile_5) = calculate_std_dev(ssimu2_total_scores)

    print(f'SSIMU2:')
    print(f'Median score:  {ssimu2_average}')
    print(f'5th Percentile:  {ssimu2_percentile_5}')
    generate_zones(ranges, ssimu2_percentile_5_total, ssimu2_average, cq, ssimu2_zones_txt_path)

match stage:
    case 0:
        workers = args.workers
        cq = int(args.quality)
        preset = args.preset
        fast_pass(src_file, output_file, tmp_dir, preset, cq, workers)
        #ranges = get_ranges(scenes_file)
        ranges, len_clip = keyframes_helper(src_file, 48, 240)
        calculate_metrics(src_file, output_file, tmp_dir, ranges, skip)
        calculate_zones(tmp_dir, ranges, cq)
    case 1:
        workers = args.workers
        cq = int(args.quality)
        preset = args.preset
        fast_pass(src_file, output_file, tmp_dir, preset, cq, workers)
    case 2:
        #ranges = get_ranges(scenes_file)
        ranges, len_clip = keyframes_helper(src_file, 48, 240)
        calculate_metrics(src_file, output_file, tmp_dir, ranges, skip)
    case 3:
        #ranges = get_ranges(scenes_file)
        ranges, len_clip = keyframes_helper(src_file, 48, 240)
        cq = int(args.quality)
        calculate_zones(tmp_dir, ranges, cq)
    case _:
        print(f"Stage argument invalid, exiting.")
        exit(-2)
