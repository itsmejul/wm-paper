# unique ids
sacct -j 2170501 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170501                              01:08:09   09:05:12 
2170501.bat+ 109711232K 487524760K   01:08:09   09:05:12 
2170501.ext+      1.50M    220952K   01:08:09   09:05:12 
# unique kps
 sacct -j 2170502 --format=JobID,MaxRSS,MaxVMSize,Elapsed,CPUTime
JobID            MaxRSS  MaxVMSize    Elapsed    CPUTime 
------------ ---------- ---------- ---------- ---------- 
2170502                              00:24:32   03:16:16 
2170502.bat+   8476952K  52088256K   00:24:32   03:16:16 
2170502.ext+      1.50M    220952K   00:24:32   03:16:16 