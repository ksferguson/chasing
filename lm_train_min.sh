epochs=$1
run_id=$2

python -u main.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 9001 --dropouti 0.4 --epochs $epochs --save PTB.pt

cp PTB.pt PTB_train_r${run_id}_e${epochs}.pt

python -u finetune.py --model QRNN --batch_size 20 --clip 0.2 --wdrop 0.1 --nhid 1550 --nlayers 4 --emsize 400 --dropouth 0.3 --seed 404 --dropouti 0.4 --epochs $epochs  --save PTB.pt

cp PTB.pt PTB_fine_r${run_id}_e${epochs}.pt

python pointer.py --model QRNN --lambdasm 0.1 --theta 1.0 --window 500 --bptt 5000 --save PTB.pt


cp PTB.pt PTB_ptr_r${run_id}_e${epochs}.pt

