import numpy as np
import scipy as sp
import pandas as pd

if __name__ == "__main__": 
    print "here"
    d = {  "SPY"   :[86.80, 86.70, 87.28],
            "AAPL" :[90.36, 94.18, 92.62]}
    print d
    df = pd.DataFrame(d)
    print df
    normed = df/df.ix[0]
    print normed
    normed['APPL'] = np.nan
    normed.fillna(value='0')
    print normed
    print normed[0:2]