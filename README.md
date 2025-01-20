只有rig mse约束
python runner.py -ddp -wn 'EXPRRig_Test' -blr 1e-4 -train_b 8 -val_b 8 -rmse 1.0  

加上rig mse和embdding约束
python runner.py -ddp -wn 'EXPRRig_Test' -blr 1e-4 -train_b 8 -val_b 8 -rmse 1.0 -fea_cos 1.0

加上rig mse和embedding 和perceptual约束
python runner.py -ddp -wn 'EXPRRig_Test' -blr 1e-4 -train_b 8 -val_b 8 -rmse 1.0 -fea_cos 1.0 -percept1.0
