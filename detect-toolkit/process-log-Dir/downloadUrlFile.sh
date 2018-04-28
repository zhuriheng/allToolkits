#!/bin/bash
set -x
if [ ! -n "$1" ]
then
    echo "must input a date eg: 20180427"
    exit
else
    echo "the input date is : "$1
fi
logDate=$1
logSaveBasePath="/workspace/data/BK/processJH_Log_Dir/logFiles/"
logSaveDir=$logSaveBasePath$logDate
# download log file
/workspace/data/softwares/qrsctlDir/qrsctlLoginAva.sh
mkdir $logSaveDir
logFile="qpulp_origin_"$logDate".json"
/workspace/data/softwares/qrsctl get qpulp-log $logFile $logSaveDir"/"$logFile
cat $logSaveDir"/"$logFile | /workspace/data/installedSoftwares/jq -r '.url' |awk '{split($0,a,"/");print "http://oquqvdmso.bkt.clouddn.com/atflow-log-proxy/images/"a[7]}' > $logSaveDir"/"$logFile"-url"


