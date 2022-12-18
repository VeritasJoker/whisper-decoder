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
SEG_TYPE := chunk
SEG_TYPE := sentence
SEG_TYPE := word
SEG_TYPE := word chunk sentence

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
%-ecog: SID := 717
%-ecog: E_LIST := $(shell seq 1 255) # 717
%-ecog: SID := 742
%-ecog: E_LIST :=  $(shell seq 1 175) # 742
%-ecog: SID := 798
%-ecog: E_LIST :=  $(shell seq 1 195) # 798

# %-ecog: E_LIST :=  $(shell seq 1 115) # 661
# %-ecog: E_LIST :=  $(shell seq 1 100) # 662
# %-ecog: E_LIST :=  $(shell seq 1 165) # 723
# %-ecog: E_LIST :=  $(shell seq 1 130) # 741
# %-ecog: E_LIST :=  $(shell seq 1 125) # 743
# %-ecog: E_LIST :=  $(shell seq 1 80) # 763

# electrode type (ifg, stg, both, all)
%-ecog: E_TYPE := ifg stg both all

# ecog paramters
%-ecog: ONSET_SHIFT := 300
%-ecog: WINDOW_SIZE := 625

%-ecog: ECOG_WINDOW_LEN := 25
%-ecog: ECOG_HOP_LEN := 5

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
		--window-len $(ECOG_WINDOW_LEN) \
		--n-fft $(N_FFT) \
		--n-mel $(N_MEL) \
		--onset-shift $(ONSET_SHIFT) \
		--window-size $(WINDOW_SIZE) \
		--save-type ecog;\

prepare-all-ecog:
	for area in $(E_TYPE); do\
		$(CMD) scripts/ecog_main.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--datum $(DATUM) \
			--elecs $(E_LIST) \
			--elec-type $$area \
			--seg-type $(SEG_TYPE) \
			--hop-len $(ECOG_HOP_LEN) \
			--window-len $(ECOG_WINDOW_LEN) \
			--n-fft $(N_FFT) \
			--n-mel $(N_MEL) \
			--onset-shift $(ONSET_SHIFT) \
			--window-size $(WINDOW_SIZE) \
			--save-type ecog;\
	done; \



##########################################################
######################### MODEL ##########################
##########################################################

# subject
%-model: SID := 742
%-model: SID := 798
%-model: SID := 717 742 798
%-model: SID := 717

# electrode list
%-model: MODEL_SIZE := tiny base small medium
%-model: MODEL_SIZE := tiny

# electrode type {ifg, stg, both, all}
%-model: ELEC_TYPE := ifg stg both
%-model: ELEC_TYPE := ifg stg both all
%-model: ELEC_TYPE := ifg

# ecog type {raw, gan}
%-model: ECOG_TYPE := raw
%-model: ONSET_SHIFT := 300

# data split (test percentage)
%-model: DATA_SPLIT = 0.1

# data split type
%-model: DATA_SPLIT_TYPE = 2-0.1
%-model: DATA_SPLIT_TYPE = 2.9-0.1
%-model: DATA_SPLIT_TYPE = 2-0.1 2.9-0.1 0.9-0.1 2-1 2.7-0.3
%-model: DATA_SPLIT_TYPE = 2.7-0.3
%-model: DATA_SPLIT_TYPE = 2-1
%-model: DATA_SPLIT_TYPE = 0.9-0.1


train-model:
	$(CMD) scripts/model_train2.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--model-size $(MODEL_SIZE) \
		--data-split $(DATA_SPLIT) \
		--data-split-type $(DATA_SPLIT_TYPE) \
		--elec-type $(ELEC_TYPE) \
		--onset-shift $(ONSET_SHIFT) \
		--ecog-type $(ECOG_TYPE) \
		--seg-type $(SEG_TYPE) \
		--saving-dir whisper-$(MODEL_SIZE)-$(SID)-$(ELEC_TYPE)-$(DATA_SPLIT_TYPE)-test$(DATA_SPLIT); \


train-all-model:
	for elec in $(ELEC_TYPE); do\
		for subject in $(SID); do\
			$(CMD) scripts/model_train.py \
				--project-id $(PRJCT_ID) \
				--sid $$subject \
				--model-size $(MODEL_SIZE) \
				--data-split $(DATA_SPLIT) \
				--data-split-type $(DATA_SPLIT_TYPE) \
				--elec-type $$elec \
				--onset-shift $(ONSET_SHIFT) \
				--ecog-type $(ECOG_TYPE) \
				--seg-type $(SEG_TYPE) \
				--saving-dir whisper-$(MODEL_SIZE)-$$subject-$$elec-$(SEG_TYPE)-$(DATA_SPLIT_TYPE)-$(ONSET_SHIFT)-test$(DATA_SPLIT); \
		done; \
	done; \


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
	python scripts/model_pred.py \
		--project-id $(PRJCT_ID) \
		--sid $(SID) \
		--seg-type $(SEG_TYPE) \
		--model-size $(MODEL_SIZE) \
		--eval-file 717_ecog_ifg_0_spec.pkl; \


pred-all-model:
	for seg in $(SEG_TYPE); do\
		$(CMD) scripts/model_pred.py \
			--project-id $(PRJCT_ID) \
			--sid $(SID) \
			--seg-type $$seg \
			--model-size $(MODEL_SIZE) \
			--eval-file 742_ecog_all_spec.pkl; \
	done; \


