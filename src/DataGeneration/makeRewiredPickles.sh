declare -a subdirs=("FF")
wrkdir=$(pwd)
rootdir="/root/GraphClassification"

for dir in ${subdirs[@]}
do
        filename=$(echo $dir | sed -e "s|/|_|g")
        DATA=$rootdir"/"$dir
        FILE=$wrkdir"/pickles/REWIRED.pkl"
        echo "submitting $DATA $FILE"
        python $wrkdir/GraphRewire.py $DATA $FILE
done
