> $2


while [ -f $1 ];
do
    nvidia-smi >> $2
    sleep 60
done
echo "Finished" >> $2