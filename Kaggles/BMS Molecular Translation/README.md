My submission for the BMS Molecular Translation Kaggle:

Models Used:
- 4 Submission Voting Ensemble(validated using RDKit)
- CNN + VIT Hybrid Encoder
- Transformer Decoder
- Trained on 2300000 samples + Finetuned on 1610000 Pseudo Samples(Didn't improve CV much) 
Post Processing:
- RDKit Ensembling
- RDKit normalization
- Batched Beam Search(Implemented from scratch in TensorFlow) on confused samples(disagreement samples between Submissions)

PUBLIC Score: 1.03 LB, 0.8 CV
