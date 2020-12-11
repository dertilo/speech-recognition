from util import data_io
from metrics_calculation import calc_wer, calc_cer

if __name__ == "__main__":
    file = "../transcriptions/hypos.txt"
    ref_file = "../transcriptions/targets.txt"
    data = [l for l in data_io.read_lines(file)]
    refdata = [l for l in data_io.read_lines(ref_file)]
    print('WER: %0.1f %%'%round(calc_wer(data,refdata)*100,2))
    print('CER: %0.1f %%'%round(calc_cer(data,refdata)*100,2))
