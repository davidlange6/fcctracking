# Extra k4run options file: per-algorithm CPU timing via Gaudi auditors.
# Usage: k4run CLDReconstruction.py /path/to/audit_timing.py --inputFiles ... -n -1
# ChronoAuditor + ChronoStatSvc print a per-algorithm table (user/system/elapsed time,
# mean and total over events) at finalize; AlgTimingAuditor adds its own summary.
from Gaudi.Configuration import INFO
from Configurables import AlgTimingAuditor, ApplicationMgr, AuditorSvc, ChronoAuditor, ChronoStatSvc

ApplicationMgr().AuditAlgorithms = True
ApplicationMgr().ExtSvc += [AuditorSvc()]
AuditorSvc().Auditors = [ChronoAuditor(), AlgTimingAuditor(OutputLevel=INFO)]
ChronoStatSvc(ChronoPrintOutTable=True, PrintUserTime=True, PrintSystemTime=True, PrintEllapsedTime=True,
              OutputLevel=INFO)
