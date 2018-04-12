#!/bin/bash
# 这个脚本的作用是将一个大数据分成 N 分，一份跑完，再跑其他的，一份一份的跑
#set -x
N=8
bash_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
inputFile=$1
inputBasePath=`dirname $inputFile`
inputFileName=`basename $inputFile`

# inputfile 
splitFilePrefix="Num-"
# split the input file by gpuNum
inputFileLineCount=`wc -l $inputFile|awk '{print $1}'`
perFileLineCount=`expr $inputFileLineCount / $N`
perFileLineCount_flag=`expr $inputFileLineCount % $N`
if [ $perFileLineCount_flag -ne 0 ]
then
    perFileLineCount=`expr $perFileLineCount + 1` 
fi
echo "perFileLineCount : "$perFileLineCount
# split 
split_output_prefix=$inputBasePath"/"$splitFilePrefix
split -l $perFileLineCount $inputFile -d -a 1 $split_output_prefix

jobNum=$[N-1]
for i in $(seq 0 $jobNum)
do
    tempDir=$inputBasePath"/job-"$i
    mkdir $tempDir
    mv $inputBasePath"/"$splitFilePrefix$i $tempDir
done
check()
{
    count=`ps -ef | grep $1 | grep -v grep |grep -v defunct |wc -l`
    #echo $count
    if [ $count -eq 0 ]; then
    	#statements
    	return 0
    fi
    return 1
}
jobFlag=0
while [ $jobFlag -lt $N ]; do
	check mp_refindeDet-res18-inference-demo.py
	if [ $? -eq 0 ]; then
		#statements
		date
		echo "no job running, so run ---"$jobFlag"--- and sleep 1800s"
		cmdStr="bash "$bash_dir"/run.sh  "$inputBasePath"/job-"$jobFlag"/"$splitFilePrefix$jobFlag
        echo $cmdStr
		$runCmd
		jobFlag=$[jobFlag+1]
		sleep 1800
	else
		date
		nowRunJobFlag=$[jobFlag-1]
		echo "job ---"$nowRunJobFlag"--- running,sleep 600s"
		sleep 600
	fi
done