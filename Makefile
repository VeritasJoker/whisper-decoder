##########################################################
############## None Configurable Parameters ##############
##########################################################
USR := $(shell whoami | head -c 2)


##########################################################
################ Configurable Parameters #################
##########################################################

PRJCT_ID := podcast
# {podcast | tfs}

############## podcast subject ##############
SID := 717
DATUM := 777_full_labels.pkl

############### Segmentation Params ###############
# segmenting word or word + context
SEG_TYPE := word_ctx
SEG_TYPE := word

############### Spectrogram Params ###############
SAMPLE_RATE := 16000
CHUNK_LEN := 30
N_FFT := 400
N_MEL := 80
AUDIO_HOP_LEN := 160
ECOG_HOP_LEN := 10


# Choose the command to run: python runs locally, echo is for debugging, sbatch
# is for running on SLURM
CMD := echo
CMD := python
CMD := sbatch submit.sh


##########################################################
######################### Audio ##########################
##########################################################

# audio file name
%-audio: AUDIO := Podcast.wav

# prepare audio (both segmentation and spectrogram)
prepare-audio:
	python scripts/audio_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--audio $(AUDIO) \
		--seg-type $(SEG_TYPE) \
		--sample-rate $(SAMPLE_RATE) \
		--chunk-len $(CHUNK_LEN) \
		--hop-len $(AUDIO_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--save-type audio_spec;\

# creates segmentations of audio file
segment-audio:
	python scripts/audio_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--audio $(AUDIO) \
		--seg-type $(SEG_TYPE) \
		--sample-rate $(SAMPLE_RATE) \
		--chunk-len $(CHUNK_LEN) \
		--hop-len $(AUDIO_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--save-type audio;\

# creates spectrograms of audio file
spec-audio:
	python scripts/audio_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--audio $(AUDIO) \
		--seg-type $(SEG_TYPE) \
		--sample-rate $(SAMPLE_RATE) \
		--chunk-len $(CHUNK_LEN) \
		--hop-len $(AUDIO_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--save-type spec;\

##########################################################
######################### ECOG ###########################
##########################################################

# electrode list
%-ecog: E_LIST := $(shell seq 1 255) # 717

# electrode type (ifg, stg, both, all)
%-ecog: E_TYPE := all

# ecog paramters
%-ecog: ONSET_SHIFT := 300
%-ecog: WINDOW_SIZE := 625

# prepare ecog data (both segmentation and spectrogram)
prepare-ecog:
	python scripts/ecog_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--elecs $(E_LIST) \
		--elec-type $(E_TYPE) \
		--seg-type $(SEG_TYPE) \
		--hop-len $(ECOG_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--onset-shift $(ONSET_SHIFT) \
		--window-size $(WINDOW_SIZE) \
		--save-type ecog_spec;\

# # creates segmentations of ecog file
segment-ecog:
	python scripts/ecog_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--elecs $(E_LIST) \
		--elec-type $(E_TYPE) \
		--seg-type $(SEG_TYPE) \
		--hop-len $(ECOG_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--onset-shift $(ONSET_SHIFT) \
		--window-size $(WINDOW_SIZE) \
		--save-type ecog;\

# creates spectrograms of ecog file
spec-ecog:
	python scripts/ecog_main.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--datum $(DATUM) \
		--elecs $(E_LIST) \
		--elec-type $(E_TYPE) \
		--seg-type $(SEG_TYPE) \
		--hop-len $(ECOG_HOP_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--onset-shift $(ONSET_SHIFT) \
		--window-size $(WINDOW_SIZE) \
		--save-type spec;\

run-slurm:
	$(CMD) scripts/model_train.py