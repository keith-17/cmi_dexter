import sys
sys.path.insert(0, r'C:\Users\maran\OneDrive\Documents\Git Profile\cmi_dexter\src')
from base_utils_qwen import SequenceExtractor

e = SequenceExtractor()
print('default', len(e.stft_extractors), len(e.cwt_extractors))
e2 = SequenceExtractor(stft_configs=None, cwt_configs=None)
print('off', len(e2.stft_extractors), len(e2.cwt_extractors))
e3 = SequenceExtractor(stft_configs=[{}], cwt_configs=[{}])
print('on', len(e3.stft_extractors), len(e3.cwt_extractors))
