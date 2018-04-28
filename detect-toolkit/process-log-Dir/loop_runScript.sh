#!/usr/bin/env bash
# 这个脚本的作用是将一个大数据分成 N 分，一份跑完，再跑其他的，一份一份的跑
set -x
if [ ! -n "$1" ]
then
    echo "must input file path"
    exit
else
    echo "the input file is : "$1
fi
N=1000
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
split -l $perFileLineCount $inputFile -d -a 4 $split_output_prefix

jobNum=$[N-1]
for i in $(seq 0 $jobNum)
do
    job_id=`printf "%.4d" $i`
    tempDir=$inputBasePath"/job-"$job_id
    mkdir $tempDir
    mv $inputBasePath"/"$splitFilePrefix$job_id  $tempDir
done
check()
{
    count=`ps auxww | grep $1 | grep -v grep |grep -v defunct |wc -l`
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
		job_id=`printf "%.4d" $jobFlag`
		echo "no job running, so run ---"$job_id"--- and sleep 100s"
		# cmdStr="bash "$bash_dir"/run.sh  "$inputBasePath"/job-"$jobFlag"/"$splitFilePrefix$jobFlag
		# $runCmd
		cd $bash_dir
		./run.sh  $inputBasePath"/job-"$job_id"/"$splitFilePrefix$job_id
		echo "nohup runing the ---"$job_id"--- and begin sleep 100s"
		jobFlag=$[jobFlag+1]
		sleep 100
	else
		date
		nowRunJobFlag=$[jobFlag-1]
		job_id=`printf "%.4d" $nowRunJobFlag`
		echo "job ---"$job_id"--- running,sleep 60s"
		sleep 60
	fi
done