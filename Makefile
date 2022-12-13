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
SEG_TYPE := word
SEG_TYPE := sentence
SEG_TYPE := chunk

############### Spectrogram Params ###############
SAMPLE_RATE := 16000
CHUNK_LEN := 30
N_FFT := 400
N_MEL := 80
AUDIO_HOP_LEN := 160


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
SID := 717
%-ecog: E_LIST := $(shell seq 1 255) # 717
SID := 742
%-ecog: E_LIST :=  $(shell seq 1 175) # 742
SID := 798
%-ecog: E_LIST :=  $(shell seq 1 195) # 798

# %-ecog: E_LIST :=  $(shell seq 1 115) # 661
# %-ecog: E_LIST :=  $(shell seq 1 100) # 662
# %-ecog: E_LIST :=  $(shell seq 1 165) # 723
# %-ecog: E_LIST :=  $(shell seq 1 130) # 741
# %-ecog: E_LIST :=  $(shell seq 1 125) # 743
# %-ecog: E_LIST :=  $(shell seq 1 80) # 763

# electrode type (ifg, stg, both, all)
%-ecog: E_TYPE := stg

# ecog paramters
%-ecog: ONSET_SHIFT := 300
%-ecog: WINDOW_SIZE := 625
%-ecog: ECOG_HOP_LEN := 10

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


##########################################################
######################### MODEL ##########################
##########################################################

# electrode list
%-model: MODEL_SIZE := tiny
%-model: MODEL_SIZE := tiny base small medium

# electrode type {ifg, stg, both, all}
%-model: ELEC_TYPE := stg

# ecog type {raw, gan}
%-model: ECOG_TYPE := raw

# data split (test percentage)
%-model: DATA_SPLIT = 0.1


train-model:
	$(CMD) scripts/model_train.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--model-size $(MODEL_SIZE) \
		--data-split $(DATA_SPLIT) \
		--elec-type $(ELEC_TYPE) \
		--ecog-type $(ECOG_TYPE) \
		--seg-type $(SEG_TYPE) \
		--saving-dir whisper-$(MODEL_SIZE)-$(SID)-$(ELEC_TYPE)-$(ECOG_TYPE)-$(SEG_TYPE)-test$(DATA_SPLIT); \


audio-model:
	$(CMD) scripts/model_audio.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--seg-type $(SEG_TYPE) \


ecog-model:
	$(CMD) scripts/model_ecog.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--elec-type $(ELEC_TYPE) \
		--ecog-type $(ECOG_TYPE) \
		--seg-type $(SEG_TYPE) \


test-model:
	python scripts/model_test.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--seg-type $(SEG_TYPE) \
		--model-size $(MODEL_SIZE) \
		--data-split $(DATA_SPLIT) \
		--elec-type $(ELEC_TYPE) \
		--ecog-type $(ECOG_TYPE) \
		--saving-dir whisper-$(MODEL_SIZE)-$(SID)-$(ELEC_TYPE)-$(ECOG_TYPE)-$(SEG_TYPE)-test$(DATA_SPLIT); \


pred-model:
	$(CMD) scripts/model_pred.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--seg-type $(SEG_TYPE) \
		--model-size $(MODEL_SIZE) \
		--eval-file 717_ecog_all_spec.pkl \
		--eval-model whisper-tiny-717-all-raw-word-test0.05; \


pred-all-model:
	for size in $(MODEL_SIZE); do\
		$(CMD) scripts/model_pred.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--seg-type $(SEG_TYPE) \
			--model-size $$size \
			--eval-file audio_spec.pkl; \
	done; \


