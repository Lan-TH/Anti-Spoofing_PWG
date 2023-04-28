# extract features
RootDir=`pwd`
echo 'Current Dir: '${RootDir}
python ./preprocess/compute_LPS.py --in_dir ./_saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/noadv_cocoder_wav \
                                   --out_dir ./_saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/noadv_cocoder_wav_feature \
                                   --param_json_path ./preprocess/conf/stft_T45.json





# train
GPU=1
python develop.py  --resume _saved/models/LA_lcnn_LPSseg_uf_seg600/20221204_192106/model_best.pth \
                    --protocol_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
                    --asv_score_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.dev.gi.trl.scores.txt \
                    --device ${GPU}    


# test
python eval.py    --resume _saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/checkpoint-epoch10.pth \
                  --protocol_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                  --asv_score_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
                  --device ${GPU}


# attack
python attack.py  --resume _saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/model_best.pth \
                  		--protocol_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
                  		--asv_score_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
                  		--device ${GPU}


# test
python eval_adv.py     --resume _saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/model_best.pth \
          						 --adv_data _saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/fgsm_adv_egs_None_${epsilon} \
 		                   --protocol_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt \
 		                   --asv_score_file ~/asvspoof2019/data_logical/LA/ASVspoof2019_LA_asv_scores/ASVspoof2019.LA.asv.eval.gi.trl.scores.txt \
 		                   --device ${GPU}	



# generate waveform
data_dir=~/_saved/models/LA_SENet12_LPSseg_uf_seg600/20221204_192106/pgd_adv_egs_None_5.0_all_wav/
    parallel-wavegan-decode \
        --checkpoint pretrained_model/train_nodev_ljspeech_parallel_wavegan.v1.long/checkpoint-1000000steps.pkl \
        --feats-scp ./feats.scp \
        --normalize-before \
        --outdir $data_dir/generated_wav


# spectral maps
python spectral_maps.py


# waveform
python waveform.py
