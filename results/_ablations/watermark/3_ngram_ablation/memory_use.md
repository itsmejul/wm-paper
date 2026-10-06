# ngram = 1
sacct -j 2170543 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime                                                                 
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170543                              01:36:59   12:55:52 
2170543.bat+  34644680K  81413980K   01:36:59   12:55:52 
2170543.ext+      1.50M    220952K   01:36:59   12:55:52 
# ngram = 2
sacct -j 2170544 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170544                              01:49:10   14:33:20 
2170544.bat+  41393252K  88114080K   01:49:10   14:33:20 
2170544.ext+      1.50M    220952K   01:49:10   14:33:20 
# ngram = 3
 sacct -j 2170545 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170545                              01:53:45   15:10:00 
2170545.bat+  28858484K 104486320K   01:53:45   15:10:00 
2170545.ext+      1.50M    220952K   01:53:45   15:10:00 
# ngram = 4
sacct -j 2170546 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170546                              01:58:02   15:44:16 
2170546.bat+  23716.50M  72451776K   01:58:02   15:44:16 
2170546.ext+      1.50M    220952K   01:58:02   15:44:16 
# ngram = 5
sacct -j 2170547 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170547                              01:57:31   15:40:08 
2170547.bat+  48862356K 104384332K   01:57:31   15:40:08 
2170547.ext+      1.50M    220952K   01:57:31   15:40:08 
# ngram = 6
sacct -j 2170548 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170548                              01:56:46   15:34:08 
2170548.bat+  52509816K 104510212K   01:56:46   15:34:08 
2170548.ext+      1.50M    220952K   01:56:46   15:34:08 