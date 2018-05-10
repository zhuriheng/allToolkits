#!/bin/bash
# 从avatest < qpulp-log中下载线上JH数据，并且提取出url
set -x
if [ ! -n "$1" ]
then
    echo "must input a date eg: 20180427"
    exit
else
    echo "the input date is : "$1
fi
logDate=$1
logSaveBasePath="/workspace/data/data/qpulp-log/"
logSaveDir=$logSaveBasePath$logDate
# download log file
/workspace/data/tools/qrsctl_dir/qrsctlLoginAvatest.sh
mkdir $logSaveDir
logFile="qpulp_origin_"$logDate".json"
qrsctl get qpulp-log $logFile $logSaveDir"/"$logFile
cat $logSaveDir"/"$logFile | jq -r '.url' |awk '{split($0,a,"/");print "http://oquqvdmso.bkt.clouddn.com/atflow-log-proxy/images/"a[7]}' > $logSaveDir"/"$logFile"-url"


