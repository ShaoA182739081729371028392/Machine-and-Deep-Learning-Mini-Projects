My Submission for the BIRDClef2021 Kaggle Challenge: using weakly supervised audio data to classify birds inside of soundscapes:


Models Used(Bagged Ensemble):
- 1 SED(Sound Event Detection) models, trained using a ResNest50-Fast 1S
- 3 PANN ResNest50 classifiers
- 4 Other Misc Models(DenseNet121, DenseNet169, DenseNet201, CNN14(PANN pretrained))


PUBLIC SCORE: 0.90 CV(Weakly Supervised), 0.68 LB: 85th Position on Public LeaderBoard.
PRIVATE SCORE: 0.62 LB, 75th Position Private Leaderboard.
