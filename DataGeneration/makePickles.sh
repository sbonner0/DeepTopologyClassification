declare -a subdirs=("BA" "ER" "FF" "RM" "SW")
wrkdir=$(pwd)

for dir in ${subdirs[@]}
do
	filename=$(echo $dir | sed -e "s|/|_|g")
	DATA=$rootdir"/"$dir
	FILE=$wrkdir"/pickles/"$filename".pkl"
	echo "submitting $DATA $FILE"
	python $wrkdir/GenFingerPrint.py $DATA $FILE
done
