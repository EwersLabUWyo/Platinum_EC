cd '/Users/waldinian/Project_Data/Platinum_EC/BB-SF/EC/7m/Converted/'
input=$(ls $PWD/* | grep 10Hz)
cd '/Users/waldinian/Work/UWyo/Research/Flux Pipeline Project/Platinum_EC/tests'
echo $PWD
output='/Users/waldinian/Project_Data/Platinum_EC/BB-SF/EC/7m/TiltCorrectionTest'
mkdir $output
outformat='pickle'
summary='/Users/waldinian/Project_Data/Platinum_EC/BB-SF/EC/7m/summary.pickle'
methods='DR TR PF CPF CPF CPF'
order='1 5 7'

python '../src/python/ecprocessor/tilt_correction_script.py' -i $input -o $output --outformat $outformat -s $summary -m $methods -n $order