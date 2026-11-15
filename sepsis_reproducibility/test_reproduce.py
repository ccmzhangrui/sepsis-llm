import unittest
import numpy as np
import pandas as pd
from reproduce import DAY_MS, rules, summarize, with_admission_outcomes

class ReplayTests(unittest.TestCase):
    def test_boundary_and_unobserved_measurement(self):
        df=pd.DataFrame(dict(sofa0=[1,1,1,1,1],sofa48=[3,1,1,2,2],
             lac0=[np.nan,4,4,np.nan,np.nan],lac48=[np.nan,4,3,np.nan,np.nan],
             hr_sd=[20,20,20,9.99,10]))
        out=rules(df)
        self.assertEqual(out.sum().tolist(),[1,1,1])
        self.assertEqual(out.any(axis=1).tolist(),[True,True,False,True,False])
    def test_death_time_relative_to_current_admission(self):
        matrix=pd.DataFrame(dict(admissionid=[10,11,12],mort28=[False,True,False],
                                  vaso_escalation=[False,False,True]))
        dates=pd.DataFrame(dict(admissionid=[10,11,12],patientid=[1,2,2],
                                admittedat=[20*DAY_MS,3*DAY_MS,5*DAY_MS],
                                dateofdeath=[40*DAY_MS,2*DAY_MS,np.nan]))
        d=with_admission_outcomes(matrix,dates)
        self.assertEqual(d.mort28.tolist(),[True,False,False])
        self.assertEqual(d.deterioration_ref.tolist(),[True,False,True])
    def test_measured_middle_windows_do_not_feed_alert(self):
        df=pd.DataFrame(dict(sofa0=[1,1],sofa48=[1,3],hr_sd=[20,20],rr_sd=[10,10],
             lac0=[2,2],lac6=[np.nan,8],lac12=[2,4],lac24=[np.nan,2],lac48=[np.nan,2],
             alert_red=[False,True],mort28=[False,True],vaso_escalation=[False,False],
             deterioration_ref=[False,True]))
        out=summarize(df)
        self.assertEqual(out['exact_alert_matches'],2)
        self.assertEqual(out['missing_lactate']['later_24h_48h'],0)
        self.assertEqual(out['missing_lactate']['all_missing_values_bracketed'],1)
        df[['lac6','lac12']]=50
        self.assertEqual(summarize(df)['exact_alert_matches'],2)
if __name__=='__main__':
    unittest.main()
