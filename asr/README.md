Preprocess takes the given transcripts, segments, timestamps, and original interviews and produces the specified segments in folders called segmented_wavs/cm and segmented_wavs/non_cm.

Segment attempts to produce timestamps (hopefully speaker diarisation) using VAD. This will eventually be used to automate the processing of more unlabelled data.

Spectrograms produces spectrograms from given wav files. CNN attempts to classify voices from these spectrograms. The approach has since changed.

Classify creates X-vector embeddings from the given utterances. It has a binary classifier (CM vs non_CM speaking) and a clustering model that identifies a general number of speakers.